#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .utils import GraphConv, MeanAggregator
from .optim_modules import BallClusterLearningLoss
import torch.nn.functional as F

class denoise_Pool_v2(nn.Module):
    def __init__(self, feat_dim, dropout_rate=0.5, denoise_topK_rate=0.5):
        super(denoise_Pool_v2, self).__init__()
        self.denoise_topK_rate = denoise_topK_rate
        self.transform_feat = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.PReLU(feat_dim), nn.Dropout(p=dropout_rate))

    def forward(self, adj, feat, dst_num):
        feat = self.transform_feat(feat)
        feat_norm = F.normalize(feat, axis=1)  # add this

        # batch index version
        pair_list = adj[:dst_num].nonzero() # only concere these nodes connectivity
        before_edge_num = len(pair_list) / dst_num

        sim_arr = (feat_norm[pair_list[:, 0]] * feat_norm[pair_list[:, 1]]).sum(dim=1)  # orig is -1~1
        sim_arr = (sim_arr + 1) / 2  # change to 0~1

        dataset_sort_flag = True
        if dataset_sort_flag:
            set0_row = sim_arr.argsort()[:int(len(sim_arr) * (1-self.denoise_topK_rate))] # 从小到大 topN 设置为0
            left_row = sim_arr.argsort()[int(len(sim_arr) * (1-self.denoise_topK_rate)):] # 后面部分要保留着
        else:
            left_row = []
            for i in range(dst_num):
                global_row_list = torch.where(pair_list[:, 0] == i)[0]
                local_set0_row = sim_arr[global_row_list, 1].argsort()[:int(len(global_row_list) * (1-self.denoise_topK_rate))]
                local_left_row = sim_arr[global_row_list, 1].argsort()[int(len(global_row_list) * (1-self.denoise_topK_rate)):]
                local_docnodeid_list = pair_list[global_row_list][:, 1].reshape(-1)
                adj[i, local_docnodeid_list[local_set0_row]] = 0
                left_row.append(global_row_list[local_left_row])
            left_row = torch.cat(left_row)

        adj[pair_list[set0_row, 0], pair_list[set0_row, 1]] = 0
        after_edge_num = len(adj[:dst_num].nonzero()) / dst_num

        degrees = torch.sum(adj, 1)
        new_adj = adj / (degrees + 1e-6) # normalilize

        new_feat = feat_norm
        return new_adj, new_feat, sim_arr, pair_list, before_edge_num, after_edge_num, left_row

class denoise_Pool_v1(nn.Module):
    def __init__(self, feat_dim, dropout_rate=0.5, denoise_topK_rate=0.5):
        super(denoise_Pool_v1, self).__init__()
        self.denoise_topK_rate = denoise_topK_rate
        self.keep_cls = nn.Sequential(nn.Linear(feat_dim, feat_dim//2), nn.PReLU(feat_dim//2), nn.Linear(feat_dim//2, 2))

    def forward(self, adj, feat, dst_num):
        # batch index version
        pair_list = adj[:dst_num].nonzero() # only concere these nodes connectivity
        before_edge_num = len(pair_list) / dst_num

        prob_arr = self.keep_cls(feat[pair_list[:, 0]] - feat[pair_list[:, 1]])
        prob_arr = F.softmax(prob_arr, dim=1)

        dataset_sort_flag = False
        if dataset_sort_flag:
            # dataset sort
            set0_row = prob_arr[:, 1].argsort()[:int(len(prob_arr) * (1-self.denoise_topK_rate))] # prob 从小到大排序的
            left_row = prob_arr[:, 1].argsort()[int(len(prob_arr) * (1-self.denoise_topK_rate)):]
            adj[pair_list[set0_row, 0], pair_list[set0_row, 1]] = 0
        else:
            left_row = []
            for i in range(dst_num):
                global_row_list = torch.where(pair_list[:, 0] == i)[0]
                local_set0_row = prob_arr[global_row_list, 1].argsort()[:int(len(global_row_list) * (1-self.denoise_topK_rate))]
                local_left_row = prob_arr[global_row_list, 1].argsort()[int(len(global_row_list) * (1-self.denoise_topK_rate)):]
                local_docnodeid_list = pair_list[global_row_list][:, 1].reshape(-1)
                adj[i, local_docnodeid_list[local_set0_row]] = 0
                left_row.append(global_row_list[local_left_row])
            left_row = torch.cat(left_row)

        after_edge_num = len(adj[:dst_num].nonzero()) / dst_num

        degrees = torch.sum(adj, 1)
        new_adj = adj / (degrees[:, None] + 1e-6) # normalilize
        new_feat = feat
        return new_adj, new_feat, prob_arr, pair_list, before_edge_num, after_edge_num, left_row

class GCN_V(nn.Module):
    def __init__(self, feature_dim, nhid, nclass, dropout=0, gpuid=0, denoise_topK_rate=0.5):
        super(GCN_V, self).__init__()
        self.pool_v1 = True

        if self.pool_v1:
            self.pool = denoise_Pool_v1(feature_dim, dropout_rate=dropout, denoise_topK_rate=denoise_topK_rate)
        else:
            self.pool = denoise_Pool_v2(feature_dim, dropout_rate=dropout, denoise_topK_rate=denoise_topK_rate)
        self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        self.nclass = nclass
        self.fc = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(nhid))
        self.bclloss = BallClusterLearningLoss(gpuid)
        self.keep_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([2., 1.]))
        self.keep_mse = torch.nn.MSELoss()

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, adj, label, idlabel = data[0], data[1], data[2], data[3]
        dst_num = len(label)
        #new_adj, new_x, prob_arr, pair_list, before_edge_num, after_edge_num, left_row = self.pool(adj, x, dst_num)
        new_adj, new_x = adj, x

        ## get pool index and loss
        #contrastive_label_arr = (idlabel[pair_list[:, 0]] == idlabel[pair_list[:, 1]]).long()
        #balance_flag = True
        #balance_flag = False
        #if balance_flag:
        #    neg_idx = torch.where(contrastive_label_arr == 0)[0]
        #    pos_idx = torch.where(contrastive_label_arr == 1)[0]
        #    pos_idx_sample = pos_idx[torch.randperm(len(pos_idx))[:len(neg_idx)]]
        #    idx_sample = torch.cat([neg_idx, pos_idx_sample])
        #    contrastive_label_arr = contrastive_label_arr[idx_sample]
        #    prob_arr = prob_arr[idx_sample]

        #if self.pool_v1:
        #    pred_label = prob_arr.topk(1, dim=1)[1].reshape(-1)
        #    acc_rate = 1.0 * pred_label.eq(contrastive_label_arr).sum().item() / len(pred_label)
        #    tp_num = len(torch.where(contrastive_label_arr[torch.where(pred_label == 1)[0]] == 1)[0])
        #    prec = 0 if len(torch.where(pred_label == 1)[0]) == 0 else 1.0 * tp_num   / len(torch.where(pred_label == 1)[0])
        #    recall = 0 if len(torch.where(contrastive_label_arr == 1)[0]) == 0 else 1.0 * tp_num / len(torch.where(contrastive_label_arr == 1)[0])
        #else:
        #    acc_rate, prec, recall = -1, -1, -1

        ## concern prec
        #left_label_arr = (idlabel[pair_list[left_row, 0]] == idlabel[pair_list[left_row, 1]]).long()
        #left_prec = 1.0 * left_label_arr.sum().item() / len(left_row)

        gcnfeat = self.conv1(new_x, new_adj)
        gcnfeat = gcnfeat[:dst_num]
        fcfeat = self.fc(gcnfeat)
        fcfeat = F.normalize(fcfeat, dim=1)

        if output_feat:
            #return fcfeat, gcnfeat, before_edge_num, after_edge_num, acc_rate, prec, recall, left_prec
            return fcfeat, gcnfeat, 0, 0, 0, 0, 0, 0

        if return_loss:
            bclloss_dict = self.bclloss(fcfeat, label)
            return bclloss_dict, torch.tensor(0.), 0, 0, 0, 0, 0, 0
            #if self.pool_v1:
            #    keep_loss = 0.5 * self.keep_loss(prob_arr, contrastive_label_arr)
            #else:
            #    keep_loss =self.keep_mse(prob_arr, contrastive_label_arr)
            #return bclloss_dict, keep_loss, before_edge_num, after_edge_num, acc_rate, prec, recall, left_prec
        return feat


def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
    model = GCN_V(feature_dim=feature_dim,
                  nhid=nhid,
                  nclass=nclass,
                  dropout=dropout)
    return model
