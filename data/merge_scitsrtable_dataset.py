import os
import cv2
import torch
import random
import pickle
import operator
from functools import reduce
from data.base_dataset import BaseDataset
from PIL import Image
from PIL import ImageFile
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from torch_geometric.data import Data
import torch_geometric.transforms as GT
from scipy.spatial import Delaunay
from xml.dom.minidom import parse
from data.shrink_polygon import shrink_polygon
from tools.bboxtohtml_ZM import BBox2HTML

ImageFile.LOAD_TRUNCATED_IMAGES = True

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz,. "
vob = {x:ind for ind, x in enumerate(alphabet)}

# 这个函数是先对单词中的每一个字母进行编码，根据上面的vob提供的字典来编的，如“012ab”就会编成[0,1,2,10,11,38,38,38,38,38]
# 长切少补，每个单词encode成 1x10 的向量
def encode_text(ins, vob, max_len = 20, default = " "):
    out = []
    sl = len(ins)
    minl = min(sl, max_len)
    for i in range(minl):
        char = ins[i]
        if char in vob:
            out.append(vob[char])
        else:
            out.append(vob[default])
    if len(out) <= max_len:
        out = out + [vob[default]] * (max_len-len(out))
    return out

class MergeSciTSRTableDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--alpha', type=float, default=3, help='the adjustment factor alpha')
        # parser.add_argument('--num_cols', type=int, default=13, help='the number of columns for classification')
        # parser.add_argument('--num_rows', type=int, default=58, help='the number of rows for classification')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.reh, self.rew = opt.load_height, opt.load_width
        self.dataset_name = opt.dataset_name
        self.data_list = []

        if opt.dataset_name == 'SciTSR':
            self.dataroot = opt.dataroot
            self.jsonfile = os.path.join(self.dataroot, "imglist.json")
            self.jsonfile1 = os.path.join(self.dataroot, "invalidlist.json")
            self.suffix = '.png'

            if os.path.exists(self.jsonfile):  # imglist.json去掉了一些有疑问的文件
                with open(self.jsonfile, "r") as read_file:
                    self.imglist = json.load(read_file)
            else:
                # lower()是大写转小写，endswith是找到以这个结束的文件，fliter()是筛选函数
                self.imglist = list(filter(lambda fn: fn.lower().endswith('.jpg') or fn.lower().endswith('.png'),
                                           os.listdir(os.path.join(self.root_path, "img"))))
                # check_all在后面，用来检查图片对应标注有没有问题，返回没问题的图片列表
                self.imglist, self.invalidlist = self.check_all()
                with open(self.jsonfile, "w") as write_file:
                    json.dump(self.imglist, write_file)
                with open(self.jsonfile1, "w") as write_file1:
                    json.dump(self.invalidlist, write_file1)
            self.graph_transform = GT.KNNGraph(k=6)

            if opt.phase == 'test':
                SciTSR_COMP_list = []
                with open('/data/cjc/SciTSR/SciTSR-COMP.list', 'r') as f_COMP:
                    for line in f_COMP.readlines():
                        SciTSR_COMP_list.append(line.strip() + self.suffix)

                self.COMP = opt.COMP
                for img_name in self.imglist:
                    if self.COMP:
                        if img_name not in SciTSR_COMP_list:
                            continue
                    self.data_list.append(img_name)
                print('The number of test tables is: ', len(self.data_list))
            else:
                for img_name in self.imglist:
                    self.data_list.append(img_name)
                print('The number of train tables is: ', len(self.data_list))

        elif opt.dataset_name == 'ICDAR2013':
            self.suffix = '.png'
            self.add_num = 1
            # self.dataroot = os.path.join(opt.dataroot, 'images')
            self.dataroot = os.path.join(opt.dataroot, 'images_new')
            self.grid_bboxes_root = os.path.join(opt.dataroot, 'grid_bboxes')
            if opt.phase == 'train':
                self.grid_rels_row_root = os.path.join(opt.dataroot, 'grid_rels_row')
                self.grid_rels_col_root = os.path.join(opt.dataroot, 'grid_rels_col')
            else:
                if not opt.Split_GT:
                    self.grid_bboxes_root = os.path.join(opt.dataroot, 'Split_Results/data_result_20220704_v8')

            if opt.phase == 'train':
                self.data_list = self.Merge_list
                self.data_list = self.data_list
            else:
                for name in os.listdir(self.grid_bboxes_root):
                    self.data_list.append(name.replace('.txt', '.png'))
            print('The number of tables is: ', len(self.data_list))

        elif opt.dataset_name == 'WTW':
            self.dataroot = opt.dataroot
            self.suffix = '.jpg'
            if self.phase == 'train':
                self.data_list = os.listdir(os.path.join(opt.dataroot, 'new_images'))
            else:
                self.data_list = os.listdir(os.path.join(opt.dataroot, 'new_images'))

        else:
            print(opt.dataset_name)
            import pdb;pdb.set_trace()

    # readlabel()作用是读取对应图片的label，包括它的structure文件，chunk文件和img文件，但这里没有读取rel文件
    def readlabel(self, idx):
        imgfn = self.data_list[idx]
        #         print('img: {}'.format(imgfn))
        # join()是组合路径，splitext()是分离扩展名，basename()是返回最后的文件名，这里返回的是图片名

        if self.dataset_name == 'SciTSR':
            imgfn = os.path.join(self.dataroot, "images", os.path.splitext(os.path.basename(imgfn))[0] + ".png")
            structfn = os.path.join(self.dataroot, "structure_original",
                                    os.path.splitext(os.path.basename(imgfn))[0] + ".json")

            if not os.path.exists(imgfn) or not os.path.exists(structfn):
                print("can't find files.")
                print(structfn)
                print(imgfn)
                return

            with open(structfn, 'r') as f:
                structure = json.load(f)

        elif self.dataset_name == 'WTW':
            imgfn = os.path.join(self.dataroot, "new_images", os.path.splitext(os.path.basename(imgfn))[0] + self.suffix)
            xmlfn = os.path.join(self.dataroot, "xml", os.path.splitext(os.path.basename(imgfn))[0] + '..xml')

            if not os.path.exists(imgfn):
                print("can't find files.")
                print(imgfn)
                return
            if not os.path.exists(xmlfn):
                print("can't find files.")
                print(xmlfn)
                return

            domTree = parse(xmlfn)
            structure = domTree.documentElement

        img = Image.open(imgfn).convert("RGB")

        return structure, img, imgfn

    def pos_feature_SciTSR(self, st, table_width, table_height, ratio_w, ratio_h):
        # x0, y0, x1, y1 = max(int(st["x0"])*0.501-5, 0), max(int(st["y0"])*0.499-10, 0), min(int(st["x1"])*0.501-5, table_width), min(int(st["y1"])*0.499-10, table_height)
        x1, y1, x3, y3 = max(int(st["x0"]), 0), max(int(st["y0"]), 0), min(int(st["x1"]), table_width), min(int(st["y1"]), table_height)
        start_row, end_row, start_col, end_col = int(st["start_row"]), int(st["end_row"]), int(st["start_col"]), int(st["end_col"])
        x_center = (x1 + x3) / 2
        y_center = (y1 + y3) / 2
        width = x3 - x1
        height = y3 - y1
        x_norm = (x1 + x3) / table_width - 1
        y_norm = (y1 + y3) / table_height - 1
        x1_roi, y1_roi = int(x1 * ratio_w), int(y1 * ratio_h)
        x3_roi, y3_roi = int(x3 * ratio_w), int(y3 * ratio_h)
        return [x1 / table_width, y1 / table_height, x3 / table_width, y1 / table_height,
                x3 / table_width, y3 / table_height, x1 / table_width, y3 / table_height,
                x_center / table_width, y_center / table_height, width / table_width, height / table_height,
                start_row, end_row, start_col, end_col,
                x1, y1, x3, y1, x3, y3, x1, y3,
                x1_roi, y1_roi, x3_roi, y3_roi]

    def pos_feature_WTW(self, x1, y1, x2, y2, x3, y3, x4, y4, startrow, endrow, startcol, endcol, table_width, table_height, ratio_w, ratio_h):
        x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
        start_row, end_row, start_col, end_col = int(startrow), int(endrow), int(startcol), int(endcol)
        x_center = (x1 + x2 + x3 + x4) / 4
        y_center = (y1 + y2 + y3 + y4) / 4
        width = (x2 + x4 - x1 - x3) / 2
        height = (y3 + y4 - y1 - y2) / 2
        x1_roi, y1_roi = int(x1 * ratio_w), int(y1 * ratio_h)
        x3_roi, y3_roi = int(x3 * ratio_w), int(y3 * ratio_h)
        return [x1 / table_width, y1 / table_height, x2 / table_width, y2 / table_height,
                x3 / table_width, y3 / table_height, x4 / table_width, y4 / table_height,
                x_center / table_width, y_center / table_height, width / table_width, height / table_height,
                start_row, end_row, start_col, end_col,
                x1, y1, x2, y2, x3, y3, x4, y4,
                x1_roi, y1_roi, x3_roi, y3_roi]

    def get_transform(self, reh, rew):
        img_transform = []
        img_transform.append(transforms.Resize((reh, rew), Image.BICUBIC))
        img_transform.append(transforms.ToTensor())
        # img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        img_transform.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return transforms.Compose(img_transform)

    def edge_sample(self, edge_index, y, knn_num, sample_num):
        edge_index = np.array(edge_index)
        y = np.array(y)
        y_up = y[None, :]
        concat_array = np.concatenate((edge_index, y_up), axis=0)

        if len(concat_array[0]) % 10 != 0:
            return edge_index, y

        vertex_split = np.hsplit(concat_array, len(concat_array[0]) / knn_num)
        result = np.array([[0], [0], [0]])

        for a_vertex in vertex_split:
            a_vertex_sort = a_vertex.T[np.lexsort(a_vertex)].T
            gt0_count = len(a_vertex_sort[2]) - sum(a_vertex_sort[2])
            gt0_array = a_vertex_sort[:, 0:gt0_count]
            gt1_array = a_vertex_sort[:, gt0_count:]

            if len(gt0_array[0]) >= sample_num and len(gt1_array[0]) >= sample_num:
                seed0 = random.sample(range(0, len(gt0_array[0])), sample_num)
                seed1 = random.sample(range(0, len(gt1_array[0])), sample_num)
            elif len(gt0_array[0]) < sample_num:
                seed0 = random.sample(range(0, len(gt0_array[0])), len(gt0_array[0]))
                seed1 = random.sample(range(0, len(gt1_array[0])), 2 * sample_num - len(gt0_array[0]))
            else:
                seed0 = random.sample(range(0, len(gt0_array[0])), 2 * sample_num - len(gt1_array[0]))
                seed1 = random.sample(range(0, len(gt1_array[0])), len(gt1_array[0]))

            sample0 = gt0_array[:, seed0]
            sample1 = gt1_array[:, seed1]
            result = np.column_stack((result, sample0))
            result = np.column_stack((result, sample1))

        # print(result)
        # result = result[:, 1:].T
        # np.random.shuffle(result)
        # result = result.T
        # print(result)
        new_edge_index = result[0:2, 1:]
        new_y = result[2, 1:]

        return new_edge_index, new_y

    def delete_edge(self, data):
        edges = data.edge_index
        edges_new1 = []
        edges_new2 = []
        for i in range(edges.size()[1]):
            if edges[0, i] < edges[1, i]:
                edges_new1.append(edges[0, i])
                edges_new2.append(edges[1, i])
        edges_new = [edges_new1, edges_new2]
        edges_new = torch.LongTensor(edges_new)
        return edges_new

    def delete_0_edge(self, edge_index, y):
        edges_new1 = []
        edges_new2 = []
        for i in range(len(y)):
            if y[i] == 1:
                edges_new1.append(edge_index[0, i])
                edges_new2.append(edge_index[1, i])
        edges_new = [edges_new1, edges_new2]
        edges_new = torch.LongTensor(edges_new)
        return edges_new

    def cal_label(self, data, tbpos):  # 根据构造的图，计算边的标注。
        edges = data.new_edge_index  # [2, 边的个数] 无向图的边是对称的，即有2条。
        y_row = []
        y_col = []
        for i in range(edges.size()[1]):
            y_row.append(self.if_same_row(edges[0, i], edges[1, i], tbpos))   # 同行判断，horizontal prediction
            y_col.append(self.if_same_col(edges[0, i], edges[1, i], tbpos))  # 同列判断，vertical prediction
        return y_row, y_col

    def if_same_row(self, si, ti, tbpos):
        ssr, ser, ssc, sec = tbpos[si][0], tbpos[si][1], tbpos[si][2], tbpos[si][3]
        tsr, ter, tsc, tec = tbpos[ti][0], tbpos[ti][1], tbpos[ti][2], tbpos[ti][3]
        if sec == tsc - 1 or tec == ssc - 1:
            if (ssr >= tsr and ser <= ter):
                return 1
            if (tsr >= ssr and ter <= ser):
                return 1
        return 0
        # ss, se = tbpos[si][0], tbpos[si][1]
        # ts, te = tbpos[ti][0], tbpos[ti][1]wwww
        # if (ss >= ts and se <= te):
        #     return 1
        # if (ts >= ss and te <= se):
        #     return 1
        # return 0

    def if_same_col(self, si, ti, tbpos):
        ssr, ser, ssc, sec = tbpos[si][0], tbpos[si][1], tbpos[si][2], tbpos[si][3]
        tsr, ter, tsc, tec = tbpos[ti][0], tbpos[ti][1], tbpos[ti][2], tbpos[ti][3]
        if ser == tsr - 1 or ter == ssr - 1:
            if (ssc >= tsc and sec <= tec):
                return 1
            if (tsc >= ssc and tec <= sec):
                return 1
        return 0
        # ss, se = tbpos[si][2], tbpos[si][3]
        # ts, te = tbpos[ti][2], tbpos[ti][3]
        # if (ss >= ts and se <= te):
        #     return 1
        # if (ts >= ss and te <= se):
        #     return 1
        # return 0

    def if_same_cell(self):
        pass

    def beta_skeleton(self, data):
        points_4 = np.array(data)
        points_10 = self.axis_extend(points_4)
        points_10 = points_10.reshape([-1, 2])

        tri = Delaunay(points_10)
        tri_edges = np.concatenate((tri.simplices[:, 0:2], tri.simplices[:, 1:], tri.simplices[:, 0:2:1]), axis=0)
        valid_edges = self.rectify_edges(tri_edges)

        # plt.triplot(points_10[:, 0], points_10[:, 1], tri.simplices)
        # plt.scatter(points_10[:, 0], points_10[:, 1], c='r')
        # plt.show()

        return valid_edges

    def axis_extend(self, x):
        x_10 = []
        for i in range(x.shape[0]):
            poly = x[i].reshape([4, 2])
            shrink_poly = shrink_polygon(poly, 0.9)
            x_10.append([[shrink_poly[0][0], shrink_poly[0][1]], [2 * shrink_poly[0][0] / 3 + shrink_poly[1][0] / 3, 2 * shrink_poly[0][1] / 3 + shrink_poly[1][1] / 3],
                         [shrink_poly[0][0] / 3 + 2 * shrink_poly[1][0] / 3, shrink_poly[0][1] / 3 + 2 * shrink_poly[1][1] / 3], [shrink_poly[1][0], shrink_poly[1][1]],
                         [(shrink_poly[1][0] + shrink_poly[2][0]) / 2, (shrink_poly[1][1] + shrink_poly[2][1]) / 2], [shrink_poly[2][0], shrink_poly[2][1]],
                         [shrink_poly[3][0] / 3 + 2 * shrink_poly[2][0] / 3, shrink_poly[3][1] / 3 + 2 * shrink_poly[2][1] / 3], [2 * shrink_poly[3][0] / 3 + shrink_poly[2][0] / 3, 2 * shrink_poly[3][1] / 3 + shrink_poly[2][1] / 3],
                         [shrink_poly[3][0], shrink_poly[3][1]], [(shrink_poly[0][0] + shrink_poly[3][0]) / 2, (shrink_poly[0][1] + shrink_poly[3][1]) / 2]])

        return np.array(x_10)

    def rectify_edges(self, all_edges):
        valid_edges = []
        for line in all_edges:
            if line[0] // 10 != line[1] // 10:
                # tmp_line = [line[0] // 10, line[1] // 10]
                if line[0] // 10 < line[1] // 10:
                    tmp_line = [line[0] // 10, line[1] // 10]
                else:
                    tmp_line = [line[1] // 10, line[0] // 10]
                valid_edges.append(tmp_line)
        valid_edges = list(set([tuple(t) for t in valid_edges]))
        return np.array(valid_edges)

        # valid_edges = np.array(valid_edges)
        # valid_edges_switch = np.array(valid_edges[:, [1, 0]])
        # edges_new = np.vstack((valid_edges, valid_edges_switch))
        # edges_new = np.array(list(set([tuple(t) for t in edges_new])))
        # return edges_new

    def __getitem__(self, index):
        structure, img, img_name = self.readlabel(index)

        w, h = img.size
        # if w >= h:
        #     rew = self.rew
        #     ratio = rew / w
        #     reh = round(h * ratio)
        # else:
        #     reh = self.reh
        #     ratio = reh / h
        #     rew = round(w * ratio)
        #
        # if reh > self.rew or rew > self.rew:
        #     import pdb;
        #     pdb.set_trace()

        reh, rew = self.reh, self.rew
        ratio_h, ratio_w = reh / h, rew / w
        img_transformation = self.get_transform(reh, rew)
        img = img_transformation(img)
        img = img.unsqueeze(0)

        table_width, table_height = w, h
        x, pos, tbpos, xtext, imgpos, plaintext, ori_x = [], [], [], [], [], [], []

        if self.dataset_name == 'SciTSR':
            structs = structure["cells"]
            structs.sort(key=lambda p: p["id"])
            for st in structs:
                try:
                    xt = self.pos_feature_SciTSR(st, table_width, table_height, ratio_w, ratio_h)
                    x.append(xt[0:12])
                    pos.append(xt[8:10])
                    tbpos.append(xt[12:16])
                    # text = reduce(operator.add, st["content"], '')
                    text = reduce(lambda x, y: x + ' ' + y, st["content"], '')
                    xtext.append(encode_text(text, vob))
                    plaintext.append(text.encode('utf-8'))
                    ori_x.append(xt[16: 24])
                    imgpos.append(xt[-4:])     # 改过
                except:
                    continue

        elif self.dataset_name == 'WTW':
            object = structure.getElementsByTagName("object")
            for i in range(len(object)):
                x1 = float(structure.getElementsByTagName("x1")[i].firstChild.data)
                y1 = float(structure.getElementsByTagName("y1")[i].firstChild.data)
                x2 = float(structure.getElementsByTagName("x2")[i].firstChild.data)
                y2 = float(structure.getElementsByTagName("y2")[i].firstChild.data)
                x3 = float(structure.getElementsByTagName("x3")[i].firstChild.data)
                y3 = float(structure.getElementsByTagName("y3")[i].firstChild.data)
                x4 = float(structure.getElementsByTagName("x4")[i].firstChild.data)
                y4 = float(structure.getElementsByTagName("y4")[i].firstChild.data)
                startrow = float(structure.getElementsByTagName("startrow")[i].firstChild.data)
                endrow = float(structure.getElementsByTagName("endrow")[i].firstChild.data)
                startcol = float(structure.getElementsByTagName("startcol")[i].firstChild.data)
                endcol = float(structure.getElementsByTagName("endcol")[i].firstChild.data)
                xt = self.pos_feature_WTW(x1, y1, x2, y2, x3, y3, x4, y4, startrow, endrow, startcol, endcol, table_width, table_height, ratio_w, ratio_h)
                x.append(xt[0:12])
                pos.append(xt[8:10])
                tbpos.append(xt[12:16])
                text = reduce(operator.add, 'None', '')
                xtext.append(encode_text(text, vob))
                plaintext.append(text.encode('utf-8'))
                ori_x.append(xt[16: 24])
                imgpos.append(xt[-4:])

        x = torch.FloatTensor(x)
        pos = torch.FloatTensor(pos)
        # 这里调用torch_geometric.data.Data来表示一张图，根据函数的解释，x是顶点特征矩阵，pos是顶点位置矩阵
        data = Data(x=x, pos=pos)

        # 调用knn建边
        # data = self.graph_transform(data)   # 构造图的连接

        # 采用β-skeleton建边
        new_edge_index = self.beta_skeleton(ori_x)
        data.edge_index = torch.LongTensor(new_edge_index.T)

        # 采用β-skeleton + axis-align建边
        # rowmatched, colmatched = BBox2HTML(np.array(ori_x), img_name).BBoxtoHTML()
        # axis_new_edge_index = np.concatenate((rowmatched, colmatched), axis=1)
        # beta_new_edge_index = self.beta_skeleton(data)
        # all_new_edge_index = np.concatenate((axis_new_edge_index.T, beta_new_edge_index), axis=0)
        # all_new_edge_index = list(set([tuple(t) for t in all_new_edge_index]))
        # index = np.lexsort([np.array(all_new_edge_index)[:, 1], np.array(all_new_edge_index)[:, 0]])
        # all_new_edge_index = np.array(all_new_edge_index)[index, :]
        # data.edge_index = torch.LongTensor(all_new_edge_index.T)

        # edge_new = self.delete_edge(data)   # 删边，大概是让无向图变成了有向图
        # data.edge_index = edge_new
        data.new_edge_index = data.edge_index

        y_row, y_col = self.cal_label(data, tbpos)

        data.y_row = torch.LongTensor(y_row)
        data.y_col = torch.LongTensor(y_col)
        data.img = img
        data.imgpos = torch.as_tensor(imgpos, dtype=torch.float32)
        # data.nodenum = torch.LongTensor([len(structs)])
        data.xtext = torch.LongTensor(xtext)
        data.name = img_name.encode('utf-8')

        return data

    def __len__(self):
        return len(self.data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def read_structure(self):
        return

    def reset(self):
        pass
