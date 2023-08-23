import sys
sys.path.append("/home/cjc/GFTE")
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

import torch
import torch.utils.data
import numpy as np
from torch_geometric.data import Data, Dataset, DataLoader, DataListLoader
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import time
import pandas as pd
from tqdm import tqdm
from util import util
import torch_geometric

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

def graph_vis(data, preds_r, preds_c):
    row_label, col_label = data.y_row.detach().cpu().numpy(), data.y_col.detach().cpu().numpy()
    new_edge_index = data.new_edge_index.detach().numpy()
    name = os.path.basename(data.name[0].decode('utf-8'))

    get_edge_index_r = delete_0_edge(preds_r, data.new_edge_index)
    get_edge_index_c = delete_0_edge(preds_c, data.new_edge_index)
    # print(get_edge_index_r.size(), get_edge_index_c.size())
    wrong_edge_index_r = delete_right_edge(preds_r, data.y_row, data.new_edge_index)
    wrong_edge_index_c = delete_right_edge(preds_c, data.y_col, data.new_edge_index)

    data.edge_index = torch.cat((get_edge_index_r, get_edge_index_c), dim=1)
    new_pos = []
    for i in range(len(data.pos)):
        new_pos.append([data.pos[i][0], 1 - data.pos[i][1]])
    data.pos = torch.FloatTensor(new_pos)

    G1 = to_networkx(data, to_undirected=True)
    nums_nodes = len(data.pos)
    pos = np.array(data.pos)
    labels = dict(zip([i for i in range(nums_nodes)], pos))
    position = {k: v for k, v in labels.items()}
    nx.draw_networkx(G1, pos=position)
    plt.savefig(os.path.join(root, 'test_result', name))
    plt.close()

    data.edge_index = torch.cat((wrong_edge_index_r, wrong_edge_index_c), dim=1)
    if data.edge_index.size()[1] != 0:
        G2 = to_networkx(data, to_undirected=True)
        nx.draw_networkx(G2, pos=position)
        plt.savefig(os.path.join(root, 'wrong_result', name))
        plt.close()

    return 1

def delete_0_edge(preds, edge_index):
    edges_new1 = []
    edges_new2 = []
    for i in range(len(preds)):
        if preds[i] == 1:
            edges_new1.append(edge_index[0, i])
            edges_new2.append(edge_index[1, i])
    edges_new = [edges_new1, edges_new2]
    edges_new = torch.LongTensor(edges_new)
    return edges_new


def delete_right_edge(preds, y, edge_index):
    edges_new1 = []
    edges_new2 = []
    for i in range(len(preds)):
        if preds[i] != y[i]:
            edges_new1.append(edge_index[0, i])
            edges_new2.append(edge_index[1, i])
    edges_new = [edges_new1, edges_new2]
    edges_new = torch.LongTensor(edges_new)
    return edges_new

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    util.init_distributed_mode(opt)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # opt.dataroot = '/home/cjc/SciTSR-master/SciTSR/test'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # evaluator = IC15Evaluator(opt)
    test_size = len(dataset)
    print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    gen_gt = False  # True
    MODE = opt.phase # 'test'  # val
    DATASET_NAME = opt.dataset_name
    model.eval()

    root = opt.dataroot
    for data in tqdm(dataset):
        torch.cuda.synchronize()

        model.set_input(data)

        cls_pred_row, cls_pred_col = model.test()
        cls_pred_row, cls_pred_col = cls_pred_row.detach().cpu().numpy(), cls_pred_col.detach().cpu().numpy()

        graph_vis(data, cls_pred_row, cls_pred_col)
