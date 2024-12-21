# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise, timemodule, xcorr_depthwise_time
from pysot.utils.location_grid import get_gt_mask, get_text_mask
from pysot.models.timesformer_pytorch.timesformer_pytorch import TimeSformer
class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
        self.timetranstformer = TimeSformer(dim=256, image_size=256, patch_size=16, num_frames=3,
                                            num_classes=1, depth=1, heads=8, dim_head=64, attn_dropout=0.1,
                                            ff_dropout=0.1)
        self.timemodule = timemodule(256, 256)
        # build car head
        self.cor_time = xcorr_depthwise_time
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)
        self.downsample_layer3_zf = nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256))
    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf, self.zf_filter = self.neck(zf)
        self.zf = zf
        self.zf_filter[1] = F.interpolate(self.zf_filter[0], [7, 7], mode="bilinear")
        self.zf_filter[2] = F.interpolate(self.zf_filter[1], [3, 3], mode="bilinear")
    def track(self, x, ID, timelist, transformer_weight_row):

        xf = self.backbone(x)
        transformer_weight_row.append(x)
        if cfg.ADJUST.ADJUST:
            xf, _ = self.neck(xf)
        xf_filter = xf

        timelist.append(xf[0])
        a = timelist[0]
        b = transformer_weight_row[0]
        if len(timelist) >= 4:
            timelist = timelist[-3:]
            transformer_weight_row = transformer_weight_row[-3:]
            timelist[0] = a
            transformer_weight_row[0] = b

        time_image_T = []
        time_feature = torch.tensor([0]).cuda()
        time_weight = torch.tensor([0]).cuda()
        if ID > 3:
           time_image_T.append(xf[0])
           time_image_T.append(timelist[-2])
           time_image_T.append(timelist[-1])
           text_mask = get_text_mask(time_image_T[0])
           mask_hint = text_mask[0].unsqueeze(1).repeat(1, xf[0].size(1), 1, 1)
           for i in range(0, 3):
               time_image_T[i] = (mask_hint * time_image_T[i]).unsqueeze(0).type(torch.FloatTensor)
           transf_input_row = torch.cat(transformer_weight_row, 0).cuda()
           tranfromerfeature_track = transf_input_row.unsqueeze(0)
           time_weight = self.timetranstformer(tranfromerfeature_track, mask=None)
           new = torch.cat(timelist, 1).cuda()
           new = new.reshape(1, -1, len(timelist), 31, 31)
           new = self.timemodule(new)
           new = new.contiguous().squeeze().permute(1, 0, 2, 3)
           a = torch.cat([new[0], new[1]], 1)
           time_feature_final = []
           for i in range(0, len(timelist)):
               time_feature_per = self.cor_time(new[i].unsqueeze(0), self.zf[0])
               time_feature_final.append(time_feature_per)
           time_feature = time_feature_final[0]+time_feature_final[-2] + time_feature_final[-1]

        features = self.xcorr_depthwise(xf[0], self.zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], self.zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)
        features = features + time_feature*time_weight

        xf_filter[1] = F.interpolate(xf_filter[0], [15, 15], mode="bilinear")
        xf_filter[2] = F.interpolate(xf_filter[1], [7, 7], mode="bilinear")
        head_features = []
        for i in range(0, 3):
            head_feature = self.xcorr_depthwise(xf_filter[i], self.zf_filter[i])
            head_features.append(head_feature)
        head_features[0] = features
        head_features[1] = self.xcorr_depthwise(head_features[0], head_features[1])
        head_features[2] = self.xcorr_depthwise(head_features[0], head_features[2])
        cls, loc, cen, filter_reg = self.car_head(head_features)
        cls = cls * filter_reg[0]
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }, timelist, transformer_weight_row

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        tranfromerfeature = search.unsqueeze(0)
        W = self.timetranstformer(tranfromerfeature, mask=None)

        # get feature
        zf = self.backbone(template)
        zf_filter = zf
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf, _ = self.neck(zf)
            xf, _ = self.neck(xf)
        search_mask = get_gt_mask(xf[0], data['bbox'])
        mask_hint = search_mask[0].unsqueeze(1).repeat(1, xf[0].size(1), 1, 1)
        time_search = mask_hint * xf[0]
        time_search = time_search.reshape(1, 256, cfg.TRAIN.BATCH_SIZE//2, 31, 31).type(torch.FloatTensor).cuda()
        #time_search = time_search.unsqueeze(0).type(torch.FloatTensor).permute(0, 2, 1, 3, 4).cuda()
        new = self.timemodule(time_search)
        new = new.contiguous().squeeze().reshape(-1, 256, 31, 31)
        time_feature = self.cor_time(new, zf[0])
        #tranfromerfeature = time_feature.unsqueeze(0)
        #w = self.timetranstformer(tranfromerfeature, mask=None)
        time_feature = time_feature.sum(dim=0).unsqueeze(0)
        features = self.xcorr_depthwise(xf[0], zf[0])
        for i in range(len(xf) - 1):
            features_new = self.xcorr_depthwise(xf[i + 1], zf[i + 1])
            features = torch.cat([features, features_new], 1)
        features = self.down(features)
        #tranfromerfeature = features.unsqueeze(0).repeat(1, 16, 1, 1, 1)
        #w = self.timetranstformer(tranfromerfeature, mask=None)
        features = features + time_feature*W

        head_features = []
        zf_filter = self.downsample_layer3_zf(zf_filter[0])
        head_features.append(features)
        head_features.append(zf_filter)
        head_features.append(xf[0])
        cls, loc, cen, filter_reg = self.car_head(head_features)
        cls = cls * filter_reg[0].sigmoid_()
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
