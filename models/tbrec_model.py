import torch
import itertools
from .base_model import BaseModel
from . import networks_LCAT
import sys, cv2, os, pickle
import torch.nn.functional as F
sys.path.append("..")
from util import util
from util import Focal_loss
from util import OHEM_loss
from torchvision.ops import sigmoid_focal_loss
import ranger
from torch.cuda import amp

class TbRecModel(BaseModel):
    """
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--rm_layers', type=str, default='', help='remove layers when load the pretrained model')
        return parser

    def __init__(self, opt):
        self.opt = opt
        BaseModel.__init__(self, self.opt)
        self.loss_names = ['rel_cls_row', 'rel_cls_col', 'rel_all']  # ['seg_row', 'seg_col', 'seg_rc', 'cls_row', 'cls_col']
        self.model_names = ['Backbone', 'CellLlocPre']  # ['Backbone','CellBboxSeg', 'CellLlocPre']

        init_mode = False
        if self.isTrain:
            init_mode = not self.opt.continue_train

        if opt.distributed:
            self.netBackbone = networks_LCAT.resnet_fpn_backbone('resnet50', init_mode, self.opt.distributed, self.opt.gpu)
            self.netCellLlocPre = networks_LCAT.cell_loc_head(self.opt.num_node_features, self.opt.vocab_size, self.opt.num_text_features,
                                                     self.opt.num_classes, self.opt.load_height, self.opt.load_width,
                                                     self.opt.alpha, self.device, self.opt.distributed, self.opt.gpu)
        else:
            self.netBackbone = networks_LCAT.resnet_fpn_backbone('resnet50', init_mode, self.opt.distributed, self.opt.gpu_ids)
            self.netCellLlocPre = networks_LCAT.cell_loc_head(self.opt.num_node_features, self.opt.vocab_size, self.opt.num_text_features,
                                                     self.opt.num_classes, self.opt.load_height, self.opt.load_width,
                                                     self.opt.alpha, self.device, self.opt.distributed, self.opt.gpu_ids)
        # print(self.netBackbone)

        if self.isTrain:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(self.device)
            # self.criterion = Focal_loss.BCEFocalLoss().to(self.device)
            # self.criterion = OHEM_loss.OHEMLoss(keep_rate=0.7).to(self.device)

            self.optimizer = torch.optim.Adam(itertools.chain(self.netBackbone.parameters(), self.netCellLlocPre.parameters()), lr=self.opt.lr, weight_decay=1e-5, betas=(opt.beta1, 0.999))
            self.optimizers.extend([self.optimizer])
            self.scaler = torch.cuda.amp.GradScaler()

    def set_input(self, input):
        self.table_names = input.name
        self.table_images = input.img.to(self.device)
        self.x = input.x.to(self.device)
        self.edge_index = input.edge_index.to(self.device)
        self.edge_pairing_index = input.new_edge_index.to(self.device)
        self.xtext = input.xtext.unsqueeze(1).to(self.device)
        # self.xtext = input.xtext.to(self.device)
        # self.nodenum = input.nodenum.to(self.device)
        self.imgpos = list(input.imgpos.unsqueeze(0).to(self.device))
        # self.row_num = input.row_num.to(self.device)
        # self.col_num = input.col_num.to(self.device)
        self.y_row = input.y_row.to(self.device)
        self.y_col = input.y_col.to(self.device)

    def forward(self):
        inter_feat = self.netBackbone(self.table_images)
        if self.isTrain:
            self.cls_row_score, self.cls_col_score = self.netCellLlocPre(x=self.x, img=inter_feat, \
                    imgpos=self.imgpos, text=self.xtext, edge_index=self.edge_index, \
                    edge_pairing_index=self.edge_pairing_index, phase=self.opt.phase)
        else:
            self.cls_row_score, self.cls_col_score = self.netCellLlocPre(x=self.x, img=inter_feat, \
                    imgpos=self.imgpos, text=self.xtext, edge_index=self.edge_index, \
                    edge_pairing_index=self.edge_pairing_index, phase=self.opt.phase)

    def backward(self):
        try:
            # CE Loss
            loss_rel_cls_row = torch.mean(self.criterion(self.cls_row_score, self.y_row))
            loss_rel_cls_col = torch.mean(self.criterion(self.cls_col_score, self.y_col))

            # Focal Loss + sigmoid
            # loss_rel_cls_row = self.criterion(self.cls_row_score[:, 1], self.y_row)
            # loss_rel_cls_col = self.criterion(self.cls_col_score[:, 1], self.y_col)

            # Focal Loss + Softmax
            # loss_rel_cls_row = self.criterion(self.cls_row_score, self.y_row)
            # loss_rel_cls_col = self.criterion(self.cls_col_score, self.y_col)

            losses = loss_rel_cls_row * 2 + loss_rel_cls_col * 4
        except:
            import pdb; pdb.set_trace()

        loss_dict = {'loss_rel_cls_row': loss_rel_cls_row, 
                     'loss_rel_cls_col': loss_rel_cls_col,
                     'loss_rel_all': losses}
        loss_dict_reduced = util.reduce_dict(loss_dict)

        self.loss_rel_cls_row = loss_dict_reduced['loss_rel_cls_row']
        self.loss_rel_cls_col = loss_dict_reduced['loss_rel_cls_col']
        self.loss_rel_all = loss_dict_reduced['loss_rel_all']
        # losses.backward()
        self.scaler.scale(losses).backward()


    def optimize_parameters(self):
        if not self.isTrain:
            self.isTrain = True
        if self.opt.distributed:
            self.netBackbone.train()
            self.netCellLlocPre.train()
        else:
            self.netBackbone.to(self.device).train()
            self.netCellLlocPre.to(self.device).train()

        self.forward()
        self.optimizer.zero_grad()
        self.backward()

        # 使用单卡调试，在loss.backward()之后optimizer.step()之前加入下面代码：
        # for name, param in self.netBackbone.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # 打印出来的参数就是没有参与loss运算的部分，他们梯度为None

        # self.optimizer.step()

        self.scaler.step(self.optimizer)
        self.scaler.update()
    
    @torch.no_grad()
    def test(self):
        if self.isTrain:
            self.isTrain = False
        if self.opt.distributed:
            self.netBackbone.eval()
            self.netCellLlocPre.eval()
        else:
            self.netBackbone.to(self.device).eval()
            self.netCellLlocPre.to(self.device).eval()

        self.forward()
        try:
            # cls_pred_row = torch.argmax(1 - F.softmax(self.cls_row_score, dim=-1), dim=-1)  # [num_cells, 2, rows_classes-1]
            cls_pred_row = torch.argmax(F.softmax(self.cls_row_score, dim=-1), dim=-1)  # [num_cells, 2, rows_classes-1]
        except:
            cls_pred_row = None

        try:
            # cls_pred_col = torch.argmax(1 - F.softmax(self.cls_col_score, dim=-1), dim=-1)  # [num_cells, 2, cols_classes-1]
            cls_pred_col = torch.argmax(F.softmax(self.cls_col_score, dim=-1), dim=-1)  # [num_cells, 2, cols_classes-1]
        except:
            cls_pred_col = None

        return cls_pred_row, cls_pred_col
