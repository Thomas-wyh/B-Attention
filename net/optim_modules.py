#coding:utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pos_loss(decision_mat, label_mat, beta, k=1):
    if label_mat.sum() == 0:
        print('cut pos 0')
        return torch.tensor(0.)
    # decision is not confidence, which is always to positive in 0~1, and not contain the class information
    # decision is a real value and the class infomation in the sign
    # this is decision mat, for pos sample, the smaller of the val the harder of the case
    decision_arr = decision_mat[label_mat].topk(k=k, largest=False)[0]
    loss = F.relu(beta - decision_arr)
    loss = loss.mean()
    return loss

def get_neg_loss(decision_mat, label_mat, beta, k=1):
    if label_mat.sum() == 0:
        print('cut neg 0')
        return torch.tensor(0.)
    # this is decision mat, for neg sample, the larger of the val, the harder of the case
    decision_arr = -1 * decision_mat[label_mat].topk(k=k, largest=True)[0]
    loss = F.relu(beta - decision_arr)
    loss = loss.mean()
    return loss

def cosine_sim(x, y):
    # return torch.mm(x, y.T)
    # return torch.matmul(x, y.T)
    return torch.matmul(x, y.transpose(-1, -2))

class ClusterLoss(nn.Module):
    def __init__(self, beta_pos=0.5, beta_neg=0.5, alpha_pos=4., alpha_neg=1., gamma_eps=0.05, losstype='allall', margin=1., pmargin=1.):
        super(ClusterLoss, self).__init__()
        #self.gamma_eps = gamma_eps
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

        self.beta_pos = nn.Parameter(torch.tensor(beta_pos))
        #self.beta_neg = nn.Parameter(torch.tensor(beta_neg))
        self.losstype = losstype
        self.margin = margin
        self.pmargin = pmargin

    def forward(self, X, labels):
        #beta_pos = F.softplus(self.beta_pos)
        #beta_neg = F.softplus(self.beta_neg)
        #beta_neg = beta_pos
        beta_pos = self.pmargin
        beta_neg = self.margin

        #X_copy = X.clone().detach()
        #decision_mat = (X.unsqueeze(1) - X_copy.unsqueeze(0)).pow(2).sum(2).sqrt()  # euclidean distance
        #decision_mat = euclidean_dist(X, X)
        decision_mat = cosine_sim(X, X)
        # label_mat = (labels.unsqueeze(0) == labels.unsqueeze(1))
        label_mat = (labels.unsqueeze(-2) == labels.unsqueeze(-1))
        #print("beta pos", beta_pos.item(), 'beta neg', beta_neg.item())
        print("losstype", self.losstype, "margin", self.margin, 'pweight', self.alpha_pos, 'pmargin', self.pmargin)

        neg_label_mat = (1-label_mat.float()).bool()
        if self.losstype == 'maxmax':
            pos_loss = get_pos_loss(decision_mat, label_mat, beta_pos, k=1)
            neg_loss = get_neg_loss(decision_mat, neg_label_mat, beta_neg, k=1)
        elif self.losstype == 'allmax':
            pos_loss = get_pos_loss(decision_mat, label_mat, beta_pos, k=label_mat.sum().item())
            neg_loss = get_neg_loss(decision_mat, neg_label_mat, beta_neg, k=1)
        elif self.losstype == 'allall':
            pos_loss = get_pos_loss(decision_mat, label_mat, beta_pos, k=label_mat.sum().item())
            neg_loss = get_neg_loss(decision_mat, neg_label_mat, beta_neg, k=neg_label_mat.sum().item())
        elif self.losstype == 'alltopk':
            pos_loss = get_pos_loss(decision_mat, label_mat, beta_pos, k=label_mat.sum().item())
            neg_loss = get_neg_loss(decision_mat, neg_label_mat, beta_neg, k=min(len(labels), neg_label_mat.sum().item()) )
        else:
            raise ValueError('loss type %s not implement'%self.lossstype)

        losses = {'ctrd_pos': pos_loss * self.alpha_pos, 'ctrd_neg': neg_loss * self.alpha_neg}
        return losses

