#coding:utf-8
import sys
from util.confidence import confidence_to_peaks
from util.deduce import peaks_to_labels
from util.evaluate import evaluate
from multiprocessing import Process, Manager
import numpy as np
import os
import torch
import torch.nn.functional as F
from util.graph import graph_propagation_onecut
from multiprocessing import Pool
from util.deduce import edge_to_connected_graph

metric_list = ['bcubed', 'pairwise', 'nmi']
def worker(param):
    i, pdict = param
    query_nodeid = ngbr_arr[i, 0]
    for j in range(1, dist_arr.shape[1]):
        doc_nodeid = ngbr_arr[i, j]
        tpl = (query_nodeid, doc_nodeid)
        dist = dist_arr[query_nodeid, j]
        if dist > cos_dist_thres:
            continue
        pdict[tpl] = dist

def format(dist_arr, ngbr_arr):
    edge_list, score_list = [], []
    for i in range(dist_arr.shape[0]):
        query_nodeid = ngbr_arr[i, 0]
        for j in range(1, dist_arr.shape[1]):
            doc_nodeid = ngbr_arr[i, j]
            tpl = (query_nodeid, doc_nodeid)
            score = 1 - dist_arr[query_nodeid, j]
            if score < cos_sim_thres:
                continue
            edge_list.append(tpl)
            score_list.append(score)
    edge_arr, score_arr = np.array(edge_list), np.array(score_list)
    return edge_arr, score_arr

def clusters2labels(clusters, n_nodes):
    labels = (-1)* np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            labels[xid.name] = ci

    cnt = len(clusters)
    idx_list = np.where(labels < 0)[0]
    for idx in idx_list:
        labels[idx] = cnt
        cnt += 1
    assert np.sum(labels<0) < 1
    return labels 

def disjoint_set_onecut(sim_dict, thres, num):
    edge_arr = []
    for edge, score in sim_dict.items():
        if score < thres:
            continue
        edge_arr.append(edge)
    pred_arr = edge_to_connected_graph(edge_arr, num)
    return pred_arr

def get_eval(cos_sim_thres):
    pred_arr = disjoint_set_onecut(sim_dict, cos_sim_thres, len(gt_arr))
    print("now is %s done"%cos_sim_thres)
    res_str = ""
    for metric in metric_list:
        res_str += evaluate(gt_arr, pred_arr, metric)
        res_str += "\n"
    return res_str

if __name__ == "__main__":
    gtfile, simfile, modelfile, outpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    gt_arr = np.load(gtfile)
    sim_dict = np.load(simfile, allow_pickle=True).item()

    thres_range = np.arange(0.60, 1.0, 0.01)  # margin 0.8

    pool = Pool(20)
    res_list = pool.map(get_eval, thres_range)
    pool.close()
    pool.join()
    for idx in range(len(thres_range)):
        sim_thres = thres_range[idx]
        res = res_list[idx]
        print('now sim thres %.8f'%sim_thres)
        print(res)
