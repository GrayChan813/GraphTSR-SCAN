import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, Sequential, GCN2Conv, GATv2Conv, norm
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch as GraphBatch
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torchvision import models
from torchvision import ops
from torchvision.ops import boxes as box_ops
import numpy as np
import cv2, os
import copy, math

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

from torch.jit.annotations import Tuple, List, Dict, Optional
from collections import OrderedDict

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 and m.affine:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, use_distributed, gpu_id, no_init=False, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    """
    if use_distributed:
        assert(torch.cuda.is_available())
        net.to(torch.device('cuda', gpu_id))
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(torch.device('cuda'))
        # net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_id], find_unused_parameters=True)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_id])
    if not no_init:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def resnet_fpn_backbone(backbone_name, pretrained, use_distributed, gpu_id, norm_layer=ops.misc.FrozenBatchNorm2d, trainable_layers=5):
    backbone = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, 
            norm_layer=ops.misc.FrozenBatchNorm2d)
    
    # select layers that wont be frozen
    assert trainable_layers <= 5 and trainable_layers >= 0
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]
    # freeze layers
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    #return_layers = {'layer2': '0', 'layer3': '1', 'layer4': '2'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256

    net = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    net = FeatureFusionForFPN(net)

    ## initalize the FeatureFusion layers
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    for submodule in net.children():
        if submodule.__class__.__name__ != "BackboneWithFPN":
            submodule.apply(init_func)
    
    return init_net(net, use_distributed, gpu_id, no_init=True)

def cell_loc_head(num_node_features, vocab_size, num_text_features, num_classes, img_h, img_w, alpha, device, use_distributed, gpu_id):
    net = Cell_Lloc_Pre(num_node_features, vocab_size, num_text_features, num_classes, img_h, img_w, alpha, device)
    return init_net(net, use_distributed, gpu_id)


##############################################################################
# Classes
##############################################################################

class OrdinalRegressionLoss(nn.Module):
    """
    """

    def __init__(self, num_class, gamma=None):
        """ 
        """
        super(OrdinalRegressionLoss, self).__init__()
        self.num_class = num_class
        self.gamma = torch.as_tensor(gamma, dtype=torch.float32)

    def _create_ordinal_label(self, gt):
        gamma_i = torch.ones(list(gt.shape)+[self.num_class-1])*self.gamma
        gamma_i = gamma_i.to(gt.device)
        gamma_i = torch.stack([gamma_i,gamma_i],-1)

        ord_c0 = torch.ones(list(gt.shape)+[self.num_class-1]).to(gt.device)
        mask = torch.zeros(list(gt.shape)+[self.num_class-1])+torch.linspace(0, self.num_class - 2, self.num_class - 1, requires_grad=False)
        mask = mask.contiguous().long().to(gt.device)
        mask = (mask >= gt.unsqueeze(len(gt.shape)))
        ord_c0[mask] = 0
        ord_c1 = 1-ord_c0
        ord_label = torch.stack([ord_c0,ord_c1],-1)
        return ord_label.long(), gamma_i

    def __call__(self, prediction, target):
        # original
        #ord_label = self._create_ordinal_label(target)
        #pred_score = F.log_softmax(prediction,dim=-1)
        #entropy = -pred_score * ord_label
        #entropy = entropy.view(-1,2,(self.num_class-1)*2)
        #loss = torch.sum(entropy, dim=-1).mean()
        # using nn.CrossEntropyLoss()
        #ord_label = self._create_ordinal_label(target)
        #criterion = nn.CrossEntropyLoss().to(ord_label.device)
        #loss = criterion(prediction, ord_label)
        # add focal
        ord_label, gamma_i = self._create_ordinal_label(target)
        pred_score = F.softmax(prediction,dim=-1)
        pred_logscore = F.log_softmax(prediction,dim=-1)
        entropy = -ord_label * torch.pow((1-pred_score), gamma_i) * pred_logscore
        entropy = entropy.view(-1,2,(self.num_class-1)*2)
        loss = torch.sum(entropy,dim=-1)
        return loss.mean()


class BackboneWithFPN(nn.Module):
    """
    copy from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/backbone_utils.py
    without extra_blocks=LastLevelMaxPool() in FeaturePyramidNetwork
    """
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(BackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x

class FeatureFusionForFPN(nn.Module):
    def __init__(self, backbone):
        super(FeatureFusionForFPN, self).__init__()
        
        self.fpn_backbone = backbone

        self.layer1_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.layer2_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
        
        self.layer3_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.layer4_bn_relu = nn.Sequential(
                        #nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth1 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth2 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

        self.smooth3 = nn.Sequential(
                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')
        return nn.functional.interpolate(x, size=(H // scale, W // scale), mode='bilinear', align_corners=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        #return F.upsample(x, size=(H, W), mode='bilinear') + y
        return nn.functional.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        fpn_outputs = self.fpn_backbone(x)
        #print(fpn_outputs['0'].shape,fpn_outputs['1'].shape,fpn_outputs['2'].shape)
        # the output of a group of fpn feature: 
        # [('0', torch.Size([1, 256, 128, 128])), 
        #  ('1', torch.Size([1, 256, 64, 64])), 
        #  ('2', torch.Size([1, 256, 32, 32])), 
        #  ('3', torch.Size([1, 256, 16, 16]))]
        layer1 = self.layer1_bn_relu(fpn_outputs['0'])
        layer2 = self.layer2_bn_relu(fpn_outputs['1'])
        layer3 = self.layer3_bn_relu(fpn_outputs['2'])
        layer4 = self.layer4_bn_relu(fpn_outputs['3'])

        fusion4_3 = self.smooth1(self._upsample_add(layer4, layer3))
        fusion4_2 = self.smooth2(self._upsample_add(fusion4_3, layer2))
        fusion4_1 = self.smooth3(self._upsample_add(fusion4_2, layer1))

        fusion4_2 = self._upsample(fusion4_2, fusion4_1)
        fusion4_3 = self._upsample(fusion4_3, fusion4_1)
        layer4 = self._upsample(layer4, fusion4_1)
        #fusion4_3 = self._upsample(fusion4_3, fusion4_2)
        #layer4 = self._upsample(layer4, fusion4_2)

        inter_feat = torch.cat((fusion4_1, fusion4_2, fusion4_3, layer4), 1) # [N, 1024, H, W]
        inter_feat = self._upsample(inter_feat, x) # [N, 1024, x_h, x_w]
        #inter_feat = torch.cat((fusion4_2, fusion4_3, layer4), 1) # [N, 1024, H, W]
        #inter_feat = self._upsample(inter_feat, x) # [N, 1024, x_h, x_w]

        return inter_feat

class Cell_Lloc_Pre(nn.Module):
    def __init__(self, num_node_feature, vocab_size, num_text_features, num_classes,
                 img_h, img_w, alpha, device, in_channels=1024):
        super(Cell_Lloc_Pre, self).__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.alpha = alpha

        # 空间特征分支
        self.gcn_pre = nn.Sequential(nn.Linear(num_node_feature, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                     nn.Linear(256, 512), nn.ReLU(inplace=True))
        
        # self.gcn_conv1 = Sequential('x, x_0, edge_index',
        #                             [(GCN2Conv(512, 512), 'x, x_0, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        # self.gcn_conv2 = Sequential('x, x_0, edge_index',
        #                             [(GCN2Conv(512, 512), 'x, x_0, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_conv1 = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_conv2 = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv1 = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv2 = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])

        self.gcn_post_row = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                     nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.gcn_post_col = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                     nn.Linear(256, 256), nn.ReLU(inplace=True))

        # 图像特征分支
        self.decode_out = nn.Sequential(
                        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True))
        self.cnn_emb = nn.Sequential(
                        nn.Linear(64*7*7, 1024), nn.ReLU(inplace=True), nn.Dropout(),
                        nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(),
                        nn.Linear(512, 512), nn.ReLU(inplace=True))
        self.gcn_conv1_img = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_conv2_img = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv1_img = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv2_img = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_post_img_row = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                              nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.gcn_post_img_col = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                              nn.Linear(256, 256), nn.ReLU(inplace=True))

        # 文本特征分支
        # self.text_embeds = nn.Embedding(vocab_size, num_text_features)
        # self.rnn_text = nn.GRU(num_text_features, 512, bidirectional=False, batch_first=True)
        self.gcn_pre_text = nn.Sequential(nn.Linear(num_text_features, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                     nn.Linear(256, 512), nn.ReLU(inplace=True))
        # self.gcn_conv1_text = Sequential('x, x_0, edge_index',
        #                             [(GCN2Conv(512, 512), 'x, x_0, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        # self.gcn_conv2_text = Sequential('x, x_0, edge_index',
        #                             [(GCN2Conv(512, 512), 'x, x_0, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        # self.gat_conv1_text = Sequential('x, edge_index',
        #                             [(GATv2Conv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        # self.gat_conv2_text = Sequential('x, edge_index',
        #                             [(GATv2Conv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_conv1_text = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_conv2_text = Sequential('x, edge_index',
                                    [(GCNConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv1_text = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gat_conv2_text = Sequential('x, edge_index',
                                    [(GATConv(512, 512), 'x, edge_index -> x'), norm.LayerNorm(512), nn.ReLU(inplace=True), ])
        self.gcn_post_text_row = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                           nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.gcn_post_text_col = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(),
                                           nn.Linear(256, 256), nn.ReLU(inplace=True))

        # 特征融合分支
        self.fc_fusion_row = nn.Sequential(*[nn.Linear(256 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(),
                                           nn.Linear(512, 256), nn.ReLU(inplace=True)])
        self.fc_fusion_col = nn.Sequential(*[nn.Linear(256 * 3, 512), nn.ReLU(inplace=True), nn.Dropout(),
                                           nn.Linear(512, 256), nn.ReLU(inplace=True)])

        self.fc_row_cls = nn.Sequential(*[nn.Linear(256 * 2, 256), nn.ReLU(inplace=True),
                                          nn.Linear(256, num_classes), nn.ReLU(inplace=True)])
        self.fc_col_cls = nn.Sequential(*[nn.Linear(256 * 2, 256), nn.ReLU(inplace=True),
                                          nn.Linear(256, num_classes), nn.ReLU(inplace=True)])

        self.w_bbox = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.w_img = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.w_text = nn.Parameter(torch.Tensor([0.5, 0.5]))

    def forward(self, x, img, imgpos, text, edge_index, edge_pairing_index, phase):
        _, _, self.img_h, self.img_w = img.shape

        # 几何特征bbox
        box_feat = self.gcn_pre(x)

        box_feat_gcn = self.gcn_conv1(box_feat, edge_index)
        box_feat_gat = self.gat_conv1(box_feat, edge_index)
        box_feat = self.w_bbox[0] * box_feat_gcn + (1 - self.w_bbox[0]) * box_feat_gat

        box_feat_gcn = self.gcn_conv2(box_feat, edge_index)
        box_feat_gat = self.gat_conv2(box_feat, edge_index)
        box_feat = self.w_bbox[1] * box_feat_gcn + (1 - self.w_bbox[1]) * box_feat_gat

        box_feat_row = self.gcn_post_row(box_feat)
        box_feat_col = self.gcn_post_col(box_feat)
        # print(self.w[0], self.w[1])

        # 图像特征
        decode_feat = self.decode_out(img)
        img_feat = ops.roi_align(decode_feat, imgpos, 7) #[num_node, 512, 7, 7]
        img_feat = self.cnn_emb(img_feat.view(img_feat.size(0), -1))  # N*512 -- 这个N是从左到右，从上到下，进行排序的

        img_feat_gcn = self.gcn_conv1_img(img_feat, edge_index)
        img_feat_gat = self.gat_conv1_img(img_feat, edge_index)
        img_feat = self.w_img[0] * img_feat_gcn + (1 - self.w_img[0]) * img_feat_gat

        img_feat_gcn = self.gcn_conv2_img(img_feat, edge_index)
        img_feat_gat = self.gat_conv2_img(img_feat, edge_index)
        img_feat = self.w_img[1] * img_feat_gcn + (1 - self.w_img[1]) * img_feat_gat
        img_feat_row = self.gcn_post_img_row(img_feat)
        img_feat_col = self.gcn_post_img_col(img_feat)
    
        # 文本特征
        # text_feat = self.text_embeds(text)
        # text_feat, _ = self.rnn_text(text_feat)
        # text_feat = text_feat[:, -1, :]
        text_feat = self.gcn_pre_text(text)

        text_feat_gcn = self.gcn_conv1_text(text_feat, edge_index)
        text_feat_gat = self.gat_conv1_text(text_feat, edge_index)
        text_feat = self.w_text[0] * text_feat_gcn + (1 - self.w_text[0]) * text_feat_gat

        text_feat_gcn = self.gcn_conv2_text(text_feat, edge_index)
        text_feat_gat = self.gat_conv2_text(text_feat, edge_index)
        text_feat = self.w_text[1] * text_feat_gcn + (1 - self.w_text[1]) * text_feat_gat
        text_feat_row = self.gcn_post_text_row(text_feat)
        text_feat_col = self.gcn_post_text_col(text_feat)

        fusion_feat_row = torch.cat((box_feat_row, img_feat_row, text_feat_row), 1)
        fusion_feat_col = torch.cat((box_feat_col, img_feat_col, text_feat_col), 1)
        # fusion_feat_row = torch.cat((box_feat_row, text_feat), 1)
        # fusion_feat_col = torch.cat((box_feat_col, text_feat), 1)
        fusion_feat_row = self.fc_fusion_row(fusion_feat_row)
        fusion_feat_col = self.fc_fusion_col(fusion_feat_col)

        # row
        if edge_pairing_index.shape[1] != 0:
            try:
                row_s = torch.index_select(fusion_feat_row, 0, edge_pairing_index[0, :])
                row_o = torch.index_select(fusion_feat_row, 0, edge_pairing_index[1, :])
                row_feat = torch.cat((row_s, row_o), 1)
                cls_row_score = self.fc_row_cls(row_feat)
                # print(cls_row_score)

                col_s = torch.index_select(fusion_feat_col, 0, edge_pairing_index[0, :])
                col_o = torch.index_select(fusion_feat_col, 0, edge_pairing_index[1, :])
                col_feat = torch.cat((col_s, col_o), 1)
                cls_col_score = self.fc_col_cls(col_feat)
            except:
                import pdb;pdb.set_trace()
        else:
            cls_row_score = None
            cls_col_score = None

        if phase == 'train':
            return cls_row_score, cls_col_score
        elif phase == 'test':
            return cls_row_score, cls_col_score
        else:
            import pdb;pdb.set_trace()

def clones(_to_clone_module, _clone_times):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(_to_clone_module) for _ in range(_clone_times)])
