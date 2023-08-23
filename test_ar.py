import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from tqdm import tqdm
from util import util
import warnings
warnings.filterwarnings("ignore")
from tools import graph_to_ar

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    util.init_distributed_mode(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    test_size = len(dataset)
    print('The number of test images = %d. Testset: %s' % (test_size, opt.dataroot))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    gen_gt = False  # True
    MODE = opt.phase # 'test'  # val
    DATASET_NAME = opt.dataset_name
    model.eval()

    n_correct_row, n_correct_col = 0, 0
    n_total_row, n_total_col = 0, 0
    n_correct_precision_row, n_correct_precision_col = 0, 0
    n_total_precision_row, n_total_precision_col = 0, 0
    n_correct_recall_row, n_correct_recall_col = 0, 0
    n_total_recall_row, n_total_recall_col = 0, 0
    for data in tqdm(dataset):
        torch.cuda.synchronize()

        model.set_input(data)
        cls_pred_row, cls_pred_col = model.test()
        row_label, col_label = data.y_row.detach().cpu().numpy(), data.y_col.detach().cpu().numpy()
        cls_pred_row, cls_pred_col = cls_pred_row.detach().cpu().numpy(), cls_pred_col.detach().cpu().numpy()

        row_label_3x = row_label * 3
        diff_row = row_label_3x - cls_pred_row
        col_label_3x = col_label * 3
        diff_col = col_label_3x - cls_pred_col

        n_correct_row = n_correct_row + (row_label == cls_pred_row).sum()
        n_total_row = n_total_row + row_label.shape[0]
        n_correct_precision_row = n_correct_precision_row + (diff_row == 2).sum()
        n_total_precision_row = n_total_precision_row + (cls_pred_row == 1).sum()
        n_correct_recall_row = n_correct_recall_row + (diff_row == 2).sum()
        n_total_recall_row = n_total_recall_row + (row_label == 1).sum()

        n_correct_col = n_correct_col + (col_label == cls_pred_col).sum()
        n_total_col = n_total_col + col_label.shape[0]
        n_correct_precision_col = n_correct_precision_col + (diff_col == 2).sum()
        n_total_precision_col = n_total_precision_col + (cls_pred_col == 1).sum()
        n_correct_recall_col = n_correct_recall_col + (diff_col == 2).sum()
        n_total_recall_col = n_total_recall_col + (col_label == 1).sum()
        # print(n_correct_col, n_total_col, n_correct_precision_col, n_total_precision_col, n_correct_recall_col, n_total_recall_col)

    accuracy_row = n_correct_row / float(n_total_row)
    precicion_row = n_correct_precision_row / float(n_total_precision_row) if n_total_precision_row != 0 else 0
    recall_row = n_correct_recall_row / float(n_total_recall_row) if n_total_recall_row != 0 else 0
    F1_row = 2 * precicion_row * recall_row / (precicion_row + recall_row) if precicion_row != 0 or recall_row != 0 else 0
    print(n_correct_row, n_total_row, n_correct_precision_row, n_total_precision_row, n_correct_recall_row, n_total_recall_row)

    accuracy_col = n_correct_col / float(n_total_col)
    precicion_col = n_correct_precision_col / float(n_total_precision_col) if n_total_precision_col != 0 else 0
    recall_col = n_correct_recall_col / float(n_total_recall_col) if n_total_recall_col != 0 else 0
    F1_col = 2 * precicion_col * recall_col / (precicion_col + recall_col) if precicion_col != 0 or recall_col != 0 else 0
    print(n_correct_col, n_total_col, n_correct_precision_col, n_total_precision_col, n_correct_recall_col, n_total_recall_col)

    accuracy = (n_correct_row + n_correct_col) / (n_total_row + n_total_col)
    precision = (n_correct_precision_row + n_correct_precision_col) / (n_total_precision_row + n_total_precision_col)
    recall = (n_correct_recall_row + n_correct_recall_col) / (n_total_recall_row + n_total_recall_col)
    F1 = 2 * precision * recall / (precision + recall)

    print('accuray_total_row: %f, precision_row: %f, recall_row: %f, F1_row: %f' % (accuracy_row, precicion_row, recall_row, F1_row))
    print('accuray_total_col: %f, precision_col: %f, recall_col: %f, F1_col: %f' % (accuracy_col, precicion_col, recall_col, F1_col))
    print('accuray_total_all: %f, precision_all: %f, recall_all: %f, F1_all: %f' % (accuracy, precision, recall, F1))

