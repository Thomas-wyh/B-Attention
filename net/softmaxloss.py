#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .utils import GraphConv, MeanAggregator
import torch.nn.functional as F


class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0):
        super(GCN_V, self).__init__()
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)

        self.nclass = nclass
        self.classifier1 = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid))
        self.classifier2 = nn.Sequential(nn.Linear(nhid, self.nclass))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, adj = data[0], data[1]
        x = self.conv1(x, adj)

        if output_feat:
            pred = 0
            graphconv_flag = False
            graphconv_flag = True
            if graphconv_flag:
                return pred, x
            else:
                return pred, self.classifier1(x) 

        # for dgl node sample
        dst_num = len(data[2])
        x = x[:dst_num]
        pred1 = self.classifier1(x)
        ##pred2 = F.normalize(pred1)
        pred = self.classifier2(pred1)

        if return_loss:
            label = data[2]
            loss = self.loss(pred, label)
            return pred, loss

        return pred


def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN_V(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
