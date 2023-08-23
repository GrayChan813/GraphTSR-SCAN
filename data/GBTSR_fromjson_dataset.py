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
# from sentence_transformers import SentenceTransformer, LoggingHandler

ImageFile.LOAD_TRUNCATED_IMAGES = True

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz,. "
vob = {x:ind for ind, x in enumerate(alphabet)}
# 这个函数是先对单词中的每一个字母进行编码，根据上面的vob提供的字典来编的，如“012ab”就会编成[0,1,2,10,11,38,38,38,38,38]
# 长切少补，每个单词encode成 1x10 的向量
def encode_text(ins, vob, max_len=10, default=" "):
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

class GBTSRFromjsonDataset(BaseDataset):
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
            if self.phase == 'train':
                with open(os.path.join(self.dataroot, 'data_list.json'), 'r') as read_file:
                    self.data_list = json.load(read_file)
            elif self.phase == 'test':
                self.COMP = opt.COMP
                if not self.COMP:
                    with open(os.path.join(self.dataroot, 'data_list.json'), 'r') as read_file:
                        self.data_list = json.load(read_file)
                else:
                    SciTSR_COMP_list = []
                    new_data_list = []
                    with open('/data/cjc/SciTSR/SciTSR-COMP.list', 'r') as f_COMP:
                        for line in f_COMP.readlines():
                            SciTSR_COMP_list.append(line.strip() + '.png')
                    with open(os.path.join(self.dataroot, 'data_list.json'), 'r') as read_file:
                        self.data_list = json.load(read_file)
                    for img_name in self.data_list:
                        if img_name in SciTSR_COMP_list:
                            new_data_list.append(img_name)
                    self.data_list = new_data_list

        elif opt.dataset_name == 'WTW':
            self.dataroot = opt.dataroot
            if self.phase == 'train':
                self.data_list = os.listdir(os.path.join(self.dataroot, 'new_images'))
            else:
                self.data_list = os.listdir(os.path.join(self.dataroot, 'new_images'))

        elif opt.dataset_name == 'Honor':
            self.dataroot = opt.dataroot
            if self.phase == 'train':
                train_list = open(os.path.join(self.dataroot, 'graph_annotations_new', 'whole_train_list.txt'), 'r')
                self.data_list = [line.strip() for line in train_list.readlines()]
            else:
                test_list = open(os.path.join(self.dataroot, 'graph_annotations_new', 'whole_test_list.txt'), 'r')
                self.data_list = [line.strip() for line in test_list.readlines()]

        elif opt.dataset_name == 'TRS':
            self.dataroot = opt.dataroot
            if self.phase == 'train':
                train_list = open(os.path.join(self.dataroot, 'graph_annotations_new_notext', 'all-line_train_list.txt'), 'r')
                self.data_list = [line.strip() for line in train_list.readlines()]
            else:
                test_list = open(os.path.join(self.dataroot, 'graph_annotations_new_notext', 'all-line_test_list.txt'), 'r')
                self.data_list = [line.strip() for line in test_list.readlines()]
        else:
            print(opt.dataset_name)
            import pdb;pdb.set_trace()

    def loadimg(self, img_path):
        if not os.path.exists(img_path):
            ("can't find files.")
            print(img_path)
            return

        img = Image.open(img_path).convert("RGB")
        return img

    def get_transform(self, reh, rew):
        img_transform = []
        img_transform.append(transforms.Resize((reh, rew), Image.BICUBIC))
        img_transform.append(transforms.ToTensor())
        # img_transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        img_transform.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        return transforms.Compose(img_transform)

    def cal_roi_pos(self, data, table_width, table_height, ratio_w, ratio_h):
        imgpos = []
        x = data.x
        for i in range(len(x)):
            x1 = min(x[i][0], x[i][2], x[i][4], x[i][6]) * table_width
            x3 = max(x[i][0], x[i][2], x[i][4], x[i][6]) * table_width
            y1 = min(x[i][1], x[i][3], x[i][5], x[i][7]) * table_height
            y3 = max(x[i][1], x[i][3], x[i][5], x[i][7]) * table_height
            x1_roi, y1_roi = int(x1 * ratio_w), int(y1 * ratio_h)
            x3_roi, y3_roi = int(x3 * ratio_w), int(y3 * ratio_h)
            imgpos.append([x1_roi, y1_roi, x3_roi, y3_roi])

        return imgpos

    def __getitem__(self, index):
        img_name = self.data_list[index]

        if self.dataset_name == 'SciTSR':
            with open(os.path.join(self.dataroot, 'graph_annotations_new', img_name.replace('.png', '.json')), 'r') as read_file:
                struct = json.load(read_file)
            img_path = struct["name"].replace('home', 'data').replace('/SciTSR-master', '')
        elif self.dataset_name == 'WTW' and self.phase == 'train':
            with open(os.path.join(self.dataroot, 'graph_annotations_new', img_name.replace('.jpg', '.json')), 'r') as read_file:
                struct = json.load(read_file)
            img_path = struct["name"].replace('home/cjc/DataSets', 'data/cjc')
        elif self.dataset_name == 'WTW' and self.phase == 'test':
            with open(os.path.join(self.dataroot, 'graph_annotations_new', img_name.replace('.jpg', '.json')), 'r') as read_file:
                struct = json.load(read_file)
            # img_path = struct["name"].replace('home/cjc/DataSets', 'data/cjc')
            img_path = struct["name"].replace('home/cjc/DataSets/WTW/test', 'data/cjc/WTW')
        elif self.dataset_name == 'Honor':
            with open(os.path.join(self.dataroot, 'graph_annotations_new', img_name.replace('.jpg', '.json')), 'r') as read_file:
                struct = json.load(read_file)
            img_path = os.path.join(self.dataroot, 'image', img_name)
        elif self.dataset_name == 'TRS':
            with open(os.path.join(self.dataroot, 'graph_annotations_new_notext', img_name.replace('.jpg', '.json')), 'r') as read_file:
                struct = json.load(read_file)
            img_path = os.path.join(self.dataroot, 'image', img_name)
        img = self.loadimg(img_path)

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

        x = torch.FloatTensor(struct["x"])
        pos = torch.FloatTensor(struct["pos"])
        data = Data(x=x, pos=pos)
        data.edge_index = torch.LongTensor(struct["edge_index"])
        data.new_edge_index = torch.LongTensor(struct["new_edge_index"])
        data.y_row = torch.LongTensor(struct["y_row"])
        data.y_col = torch.LongTensor(struct["y_col"])

        # xtext = []
        # for _ in struct["xtext"]:
        #     text = reduce(operator.add, 'none', '')
        #     xtext.append(encode_text(text, vob))
        # data.xtext = torch.LongTensor(xtext)

        data.xtext = torch.FloatTensor(struct["xtext"])
        data.name = struct["name"].encode('utf-8')
        data.text = struct["text"]

        imgpos = self.cal_roi_pos(data, w, h, ratio_w, ratio_h)
        data.img = img
        data.imgpos = torch.as_tensor(imgpos, dtype=torch.float32)

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
