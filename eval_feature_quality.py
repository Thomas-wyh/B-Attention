#coding:utf-8 
import sys
import glob
import os
import numpy as np
# from utils.misc import clusters_pickle2txts
from util.data_analysis import eval_and_plot, cluster_size_hist


#coding:utf-8
import sys
import numpy as np
import torch


# def eval_func(distmat, q_pids, g_pids, max_rank=50):
def eval_func(q_pids, g_pids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    matches = (g_pids == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        # q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # order = indices[q_idx]
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        # keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def generate_edge_pred_and_label(sim_dict, label_arr):
    edge_pred = []
    edge_label = []
    for edge, score in sim_dict.items():
        if edge[0] == edge[1]:
            continue
        edge_pred.append(score)
        edge_label.append(label_arr[edge[0]] == label_arr[edge[1]])
    
    edge_pred_arr = np.array(edge_pred)
    edge_label_arr = np.array(edge_label).astype('int32')

    return edge_pred_arr, edge_label_arr

def visualize_roc(predict, label, prefix):

    # Load predicts, labels and annos
    pred_scores = []
    # gt_scores = []
    # score_errors = []
    # score_errors_abs = []
    labels = []
    annos = []
    colors = ['tab:blue']
    lines = ['solid']
    subbox = [0.2, 0.2, 0.5, 0.5]
    zoomed_x = [0.0,0.2]
    zoomed_y = [0.8,1.0]
    # bins = []
        
    anno = ''
    pred_scores.append(predict)
    # gt_scores.append(gt_score)
    # score_err = predict - gt_score
    # score_err_abs = np.abs(score_err)
    # score_errors.append(score_err)
    # score_errors_abs.append(score_err_abs)
    labels.append(label)
    annos.append(anno)
    # bins.append('auto')
    # bins.append(100)

    #eval_and_plot(labels, pred_scores, annos, prefix)
    eval_and_plot(labels, pred_scores, annos, colors, lines, prefix, subbox, zoomed_x, zoomed_y)
    # err_prefix = os.path.join(predicts_prefix,'err_hist')
    # gt_prefix = os.path.join(predicts_prefix,'gt_hist')
    # pred_prefix = os.path.join(predicts_prefix,'pred_hist')
    # cluster_size_hist(score_errors_abs, bins, annos, err_prefix, x_label='Score Errors')
    # cluster_size_hist(gt_scores, bins, annos, gt_prefix, x_label='Ground-truth Scores')
    # cluster_size_hist(pred_scores, bins, annos, pred_prefix, x_label='Predicted Scores')

    # TODO: Add functions to show score distributions 
    # and the cluster_size distributions with a certain range of sccores


if __name__ == "__main__":
    nbI_file, nbD_file, edge_file, label_file, prefix = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

    nbI_arr = np.load(nbI_file)
    nbD_arr = np.load(nbD_file)
    edge_dict = np.load(edge_file, allow_pickle=True).item()
    label_arr = np.load(label_file)

    print(">>> File loading completed ...")

    distmat = nbD_arr[:,1:]
    q_pids = label_arr[nbI_arr[:,0]]
    g_pids = label_arr[nbI_arr[:,1:]]

    print(">>> Evaluate feature quality ...")
    # cmc, mAP = eval_func(distmat, q_pids, g_pids)
    cmc, mAP = eval_func(q_pids, g_pids)

    print("rank1:", cmc[0])
    print("mAP:", mAP)

    print(">>> Visualize ROC ...")
    edge_pred, edge_label = generate_edge_pred_and_label(edge_dict, label_arr)
    np.savez("%s/roc_pred_and_label.npz"%prefix, pred_scores=edge_pred, gt_scores=edge_label)
    visualize_roc(edge_pred, edge_label, prefix)

    print(">>> All Done !!!")
