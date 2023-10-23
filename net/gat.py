#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
#from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import SAGEConv
from .optim_modules import ClusterLoss
import torch.nn.functional as F

class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0, losstype='allall', margin=1., pweight=4., pmargin=1.0):
        super(GCN_V, self).__init__()

        #self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0., attn_drop=0.5, \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0.6, attn_drop=0.6, \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0., attn_drop=0., \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=8, feat_drop=0., attn_drop=0., \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.sage = SAGEConv(feature_dim, nhid, aggregator_type='pool', activation=F.relu)
        self.sage1 = SAGEConv(feature_dim, nhid, aggregator_type='gcn', activation=F.relu)
        self.sage2 = SAGEConv(nhid, nhid, aggregator_type='gcn', activation=F.relu)

        self.nclass = nclass
        self.fc = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid))
        self.loss = torch.nn.MSELoss()
        #self.bclloss = BallClusterLearningLoss()
        self.bclloss = ClusterLoss(losstype=losstype, margin=margin, alpha_pos=pweight, pmargin=pmargin)

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, block_list, label, idlabel = data[0], data[1], data[2], data[3]

        # layer1
        gcnfeat = self.sage1(block_list[0], x)
        gcnfeat = F.normalize(gcnfeat, p=2, dim=1)

        # layer2
        gcnfeat = self.sage2(block_list[1], gcnfeat)
        # layer3
        fcfeat = self.fc(gcnfeat)
        fcfeat = F.normalize(fcfeat, dim=1)

        if output_feat:
            return fcfeat, gcnfeat

        if return_loss:
            bclloss_dict = self.bclloss(fcfeat, label)
            return bclloss_dict

        return fcfeat


#def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
#    model = GCN_V(feature_dim=feature_dim,
#                  nhid=nhid,
#                  nclass=nclass,
#                  dropout=dropout)
#    return model
