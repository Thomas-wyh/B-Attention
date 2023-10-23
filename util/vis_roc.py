#coding:utf-8 
import sys
import glob
import os
import numpy as np
# from utils.misc import clusters_pickle2txts
from data_analysis import eval_and_plot, cluster_size_hist


# Ablation: fusion methods
anno_ablation_fusion = {'Attn2': r'$\theta_{ctx} \mathbf{A}_{ctx} + \theta_{self} \mathbf{A}_{self}$',
                        'mulMap': r'$\mathbf{A}_{ctx} \odot \mathbf{A}_{self}$',
                        'sumMap': r'$\mathbf{A}_{ctx} + \mathbf{A}_{self}$'
}
color_ablation_fusion={'Attn2': 'tab:blue',
                        'mulMap': 'tab:orange',
                        'sumMap': 'tab:green'
}
line_ablation_fusion={'Attn2': 'solid',
                        'mulMap': 'solid',
                        'sumMap': 'solid'
}

# Ablation: attention topological architectures
anno_ablation_ctx_topologies = {'self-attention': r'$\mathbf{A}_{self}$',
                        'Attn_ctx_dash_with_W_ctx': r'$\widetilde{\mathbf{A}}_{ctx}\ \mathrm{w/}\  W_{ctx}$',
                        'Attn_ctx_dash_without_W_ctx': r'$\widetilde{\mathbf{A}}_{ctx}\ \mathrm{w/o}\  W_{ctx}$',
                        'Attn_ctx_with_W_ctx': r'$\mathbf{A}_{ctx}\ \mathrm{w/}\  W_{ctx}$',
                        'Attn_ctx_without_W_ctx': r'$\mathbf{A}_{ctx}\ \mathrm{w/o}\  W_{ctx}$',
                        'Attn2_dash_with_W_ctx': r'$\widetilde{\mathbf{Attn^2}}\ \mathrm{w/}\  W_{ctx}$',
                        'Attn2_dash_without_W_ctx': r'$\widetilde{\mathbf{Attn^2}}\ \mathrm{w/o}\  W_{ctx}$',
                        'Attn2_with_W_ctx': r'$\mathbf{Attn^2}\ \mathrm{w/}\  W_{ctx}$',
                        'Attn2_without_W_ctx': r'$\mathbf{Attn^2}\ \mathrm{w/o}\  W_{ctx}$'
}
color_ablation_ctx_topologies={'self-attention': 'tab:purple',
                        'Attn_ctx_dash_with_W_ctx': 'tab:orange',
                        'Attn_ctx_dash_without_W_ctx': 'tab:orange',
                        'Attn_ctx_with_W_ctx': 'tab:green',
                        'Attn_ctx_without_W_ctx': 'tab:green',
                        'Attn2_dash_with_W_ctx': 'tab:red',
                        'Attn2_dash_without_W_ctx': 'tab:red',
                        'Attn2_with_W_ctx': 'tab:blue',
                        'Attn2_without_W_ctx': 'tab:blue' 
}
line_ablation_ctx_topologies={'self-attention': 'solid',
                        'Attn_ctx_dash_with_W_ctx': 'solid',
                        'Attn_ctx_dash_without_W_ctx': 'dashed',
                        'Attn_ctx_with_W_ctx': 'solid',
                        'Attn_ctx_without_W_ctx': 'dashed',
                        'Attn2_dash_with_W_ctx': 'solid',
                        'Attn2_dash_without_W_ctx': 'dashed',
                        'Attn2_with_W_ctx': 'solid',
                        'Attn2_without_W_ctx': 'dashed' 
}


# Effectiveness: Celeb part1-9
anno_effectiveness_part1_9={'Part1': 'Part1',
                        'Part3': 'Part3',
                        'Part5': 'Part5',
                        'Part7': 'Part7',
                        'Part9': 'Part9'                 
}

color_effectiveness_part1_9={'Part1': 'tab:blue',
                        'Part3': 'tab:orange',
                        'Part5': 'tab:green',
                        'Part7': 'tab:red',
                        'Part9': 'tab:purple'                 
}

line_effectiveness_part1_9={'Part1': 'solid',
                        'Part3': 'solid',
                        'Part5': 'solid',
                        'Part7': 'solid',
                        'Part9': 'solid'                 
}


# Effectiveness: different benchmarks
anno_effectiveness_different_benchmarks={'Original': 'Original',
                        'GCN': 'GCN',
                        'GCN_self': 'GCN w/ 'r'$\mathbf{A}_{self}$',
                        'GCN_Attn2': 'GCN w/ 'r'$\mathbf{Attn^2}$',
                        'Trans_self': 'Trans',
                        'Trans_Attn2': 'Trans w/ 'r'$\mathbf{Attn^2}$'
}

color_effectiveness_different_benchmarks={'Original': 'tab:brown',
                        'GCN': 'tab:olive',
                        'GCN_self': 'tab:purple',
                        'GCN_Attn2': 'tab:blue',
                        'Trans_self': 'tab:orange',
                        'Trans_Attn2': 'tab:green'
}

line_effectiveness_different_benchmarks={'Original': 'solid',
                        'GCN': 'solid',
                        'GCN_self': 'solid',
                        'GCN_Attn2': 'solid',
                        'Trans_self': 'solid',
                        'Trans_Attn2': 'solid'
}


anno_dicts = {'ablation_fusion':anno_ablation_fusion,
                'ablation_ctx_topologies':anno_ablation_ctx_topologies,
                'performance_celeb_part1-9_GCN_Attn2':anno_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN_self':anno_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN':anno_effectiveness_part1_9,
                'performance_celeb_part1':anno_effectiveness_different_benchmarks,
                'performance_deepfashion':anno_effectiveness_different_benchmarks,
                'performance_msmt':anno_effectiveness_different_benchmarks,
                'performance_veri':anno_effectiveness_different_benchmarks
}


color_dicts = {'ablation_fusion':color_ablation_fusion,
                'ablation_ctx_topologies':color_ablation_ctx_topologies,
                'performance_celeb_part1-9_GCN_Attn2':color_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN_self':color_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN':color_effectiveness_part1_9,
                'performance_celeb_part1':color_effectiveness_different_benchmarks,
                'performance_deepfashion':color_effectiveness_different_benchmarks,
                'performance_msmt':color_effectiveness_different_benchmarks,
                'performance_veri':color_effectiveness_different_benchmarks
}


line_dicts = {'ablation_fusion':line_ablation_fusion,
                'ablation_ctx_topologies':line_ablation_ctx_topologies,
                'performance_celeb_part1-9_GCN_Attn2':line_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN_self':line_effectiveness_part1_9,
                'performance_celeb_part1-9_GCN':line_effectiveness_part1_9,
                'performance_celeb_part1':line_effectiveness_different_benchmarks,
                'performance_deepfashion':line_effectiveness_different_benchmarks,
                'performance_msmt':line_effectiveness_different_benchmarks,
                'performance_veri':line_effectiveness_different_benchmarks
}

subbox_dict = {'ablation_fusion':[0.4, 0.38, 0.5, 0.5],
                'ablation_ctx_topologies':[0.2, 0.2, 0.45, 0.45],
                'performance_celeb_part1-9_GCN_Attn2':[0.25, 0.2, 0.5, 0.5],
                'performance_celeb_part1-9_GCN_self':[0.25, 0.2, 0.5, 0.5],
                'performance_celeb_part1-9_GCN':[0.25, 0.2, 0.5, 0.5],
                'performance_celeb_part1':[0.2, 0.2, 0.45, 0.45],
                'performance_deepfashion':[0.2, 0.2, 0.45, 0.45],
                'performance_msmt':[0.2, 0.17, 0.45, 0.45],
                'performance_veri':[0.25, 0.18, 0.4, 0.4]
}

zoomed_x_dict = {'ablation_fusion':[0, 0.15],
                'ablation_ctx_topologies':[0, 0.2],
                'performance_celeb_part1-9_GCN_Attn2':[0, 0.1],
                'performance_celeb_part1-9_GCN_self':[0.05, 0.15],
                'performance_celeb_part1-9_GCN':[0, 0.1],
                'performance_celeb_part1':[0, 0.2],
                'performance_deepfashion':[0, 0.2],
                'performance_msmt':[0.0, 0.2],
                'performance_veri':[0.0, 0.2]
}

zoomed_y_dict = {'ablation_fusion':[0.85, 1.0],
                'ablation_ctx_topologies':[0.8, 1.0],
                'performance_celeb_part1-9_GCN_Attn2':[0.9, 1.0],
                'performance_celeb_part1-9_GCN_self':[0.85, 0.95],
                'performance_celeb_part1-9_GCN':[0.82, 0.92],
                'performance_celeb_part1':[0.8, 1.0],
                'performance_deepfashion':[0.8, 1.0],
                'performance_msmt':[0.8, 1.0],
                'performance_veri':[0.8, 1.0]
}



if __name__ == "__main__":
    predicts_prefix = sys.argv[1]

    #curve_type= predicts_prefix.split()[-1]
    _, curve_type= os.path.split(predicts_prefix)

    print(curve_type)

    # anno_dict = anno_ablation_fusion
    # color_dict = color_ablation_fusion
    # line_dict = line_ablation_fusion

    anno_dict = anno_dicts[curve_type]
    color_dict = color_dicts[curve_type]
    line_dict = line_dicts[curve_type]

    # # Load labels
    # labels = np.load(labelfile)

    # Load predicts, labels and annos
    pred_scores = []
    gt_scores = []
    score_errors = []
    score_errors_abs = []
    labels = []
    annos = []
    colors = []
    lines = []
    subbox = subbox_dict[curve_type]
    zoomed_x = zoomed_x_dict[curve_type]
    zoomed_y = zoomed_y_dict[curve_type]
    # bins = []
    for filename in glob.glob('%s/*.npz'%predicts_prefix):
        data = np.load(filename)
        predict = data['pred_scores']
        gt_score = data['gt_scores']
        # label = gt_score >= 1.0
        label = gt_score >= 0.8
        label = label.astype('int32')
        anno = os.path.splitext(os.path.basename(filename))[0]
        pred_scores.append(predict)
        gt_scores.append(gt_score)
        score_err = predict - gt_score
        score_err_abs = np.abs(score_err)
        score_errors.append(score_err)
        score_errors_abs.append(score_err_abs)
        labels.append(label)
        annos.append(anno_dict[anno])
        colors.append(color_dict[anno])
        lines.append(line_dict[anno])

        # bins.append('auto')
        # bins.append(100)

    # eval_and_plot(labels, pred_scores, annos, colors, lines, predicts_prefix)
    eval_and_plot(labels, pred_scores, annos, colors, lines, predicts_prefix, subbox, zoomed_x, zoomed_y)
    err_prefix = os.path.join(predicts_prefix,'err_hist')
    gt_prefix = os.path.join(predicts_prefix,'gt_hist')
    pred_prefix = os.path.join(predicts_prefix,'pred_hist')
    #cluster_size_hist(score_errors_abs, bins, annos, err_prefix, x_label='Score Errors')
    #cluster_size_hist(gt_scores, bins, annos, gt_prefix, x_label='Ground-truth Scores')
    #cluster_size_hist(pred_scores, bins, annos, pred_prefix, x_label='Predicted Scores')

    # TODO: Add functions to show score distributions 
    # and the cluster_size distributions with a certain range of sccores
