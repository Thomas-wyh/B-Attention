#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

#import matplotlib
#matplotlib.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Inputs: labels (binary labels, i.e., local labels); predictions (predicted scores)
# Return: fpr, tpr
def generate_roc_points(labels, predictions):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    return fpr, tpr


def plot_figures(fpr, tpr, annos, colors, lines, work_dir=None, subbox=None, zoomed_x=None, zoomed_y=None):
    if work_dir is not None:
        roc_fn = os.path.join(work_dir, 'roc.png')
        roc_zoomed_fn = os.path.join(work_dir, 'roc_zoomed.png')
    else:
        roc_fn = 'roc.png'
        roc_zoomed_fn = 'roc_zoomed.png'

    plt.rc('font', size=12)
    # Plot a figure showing the entire curves
    plt.figure(1)
    #plt.xlim(0, 0.2)
    #plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    for i in range(len(annos)):
        # plt.plot(fpr[i], tpr[i], label=annos[i])
        plt.plot(fpr[i], tpr[i], label=annos[i], color=colors[i], linestyle=lines[i])

    if subbox is None:
        plt.xlabel('Average Edge Noise Rate (ENR)')
        #plt.xlabel('Average Negtive to Positive Neighbour Rate (NPR)')
        plt.ylabel('mAP')
    else:
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
    #plt.title('ROC curve')
    plt.legend(loc='best')

    if subbox is not None:
        plt.axhspan(zoomed_y[0], zoomed_y[1], zoomed_x[0]+0.03, zoomed_x[1]+0.03, facecolor='lightgray', edgecolor='gray')
        #line1_x, line1_y = [0.2, 0.8], [1.0, 0.8]    
        #line2_x, line2_y = [0.0, 0.3], [0.8, 0.3]    
        #plt.plot(line1_x, line1_y, color='gray')
        #plt.plot(line2_x, line2_y, color='gray')

        #line_x, line_y = [0.15, 0.22], [0.82, 0.6]    
        # line_x, line_y = [0.2, 0.25], [0.75, 0.7]    
        line_x, line_y = [zoomed_x[1], zoomed_x[1]+0.05], [zoomed_y[0]-0.05, zoomed_y[0]-0.05-0.05]    
        #plt.plot(line_x, line_y, color='gray')
        plt.quiver(line_x[0], line_y[0], line_x[1]-line_x[0], line_y[1]-line_y[0], scale_units='xy', angles='xy', scale=1, color='gray')
        #plt.arrow(line_x[-1], line_y[-1], dx=0.05, dy=0.05, color='gray')

        # Plot a zoomed region in the original one
        plt.axes(subbox, facecolor='lightgray')
        # n, bins, patches = plt.hist(s, 400, normed=1)
        # plt.title('Probability')
        plt.xlim(zoomed_x[0], zoomed_x[1])
        plt.ylim(zoomed_y[0], zoomed_y[1])
        plt.xticks([])
        plt.yticks([])
        #plt.box(on=False)
        # plt.plot([0, 1], [0, 1], 'k--')
        for i in range(len(annos)):
            plt.plot(fpr[i], tpr[i], label=annos[i], color=colors[i], linestyle=lines[i])

    plt.tight_layout()
    # plt.show()
    plt.savefig(roc_fn)

    # Plot a zoomed version of curves
    if zoomed_x is not None:
        plt.figure(2)
        plt.xlim(zoomed_x[0], zoomed_x[1])
        plt.ylim(zoomed_y[0], zoomed_y[1])
        # plt.plot([0, 1], [0, 1], 'k--')
        for i in range(len(annos)):
            plt.plot(fpr[i], tpr[i], label=annos[i])
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        # plt.title('ROC curve (zoomed in at top left)')
        plt.legend(loc='best')

        plt.tight_layout()
        # plt.show()
        plt.savefig(roc_zoomed_fn)


def eval_and_plot(labels, predicts, annos, colors, lines, work_dir=None, subbox=[0.35, 0.35, 0.4, 0.4], zoomed_x=[0, 0.2], zoomed_y=[0.8, 1.0]):
    fpr = []
    tpr = []
    # for predictions in predicts:
    for i in range(len(annos)):
        fpr_, tpr_ = generate_roc_points(labels[i], predicts[i])
        fpr.append(fpr_)
        tpr.append(tpr_)

    plot_figures(fpr, tpr, annos, colors, lines, work_dir, subbox, zoomed_x, zoomed_y)


def plot_single_hist(data, bins, anno, work_dir=None, x_label='Cluster Size'):
    base_fn = '{}.png'.format(anno)

    if work_dir is not None:
        hist_fn = os.path.join(work_dir, base_fn)
    else:
        hist_fn = base_fn

    fig = plt.figure()
    # plt.hist(data, bins = bins)
    # n, bins_, patches = plt.hist(data, bins=bins, density=True, facecolor='g', alpha=0.75)
    n, bins_, patches = plt.hist(data, bins=bins, facecolor='g', alpha=0.75)
    #n, bins_, patches = plt.hist(data, bins=bins, facecolor='g', alpha=0.75)
    # plt.hist(data, bins=bins, density=True, facecolor='g', alpha=0.75)
    # plt.hist(data, bins=bins, density=True, facecolor='g', alpha=0.75)
    # plt.hist(data, bins=bins, facecolor='g', alpha=0.75)

    plt.xlabel(x_label)
    # plt.ylabel('Probability')
    # plt.title('Histogram of Cluster Size')
    # plt.xlim(0, bins.max())
    # plt.ylim(0, n.max())
    plt.grid(True)
    # plt.show()
    plt.savefig(hist_fn)
    plt.close(fig)


def plot_multiple_hist(data, bins, annos, work_dir=None, x_label='Cluster Size'):
    base_fn = 'mix.png'

    if work_dir is not None:
        hist_fn = os.path.join(work_dir, base_fn)
    else:
        hist_fn = base_fn

    fig = plt.figure()
    # kwargs = dict(histtype='stepfilled', alpha=0.2, density=True)
    kwargs = dict(histtype='stepfilled', alpha=0.2)
    for i in range(len(annos)):
        n, bins_, patches = plt.hist(data[i], bins=bins[i], label=annos[i], **kwargs)
        #n, bins_, patches = plt.hist(data[i], bins='auto', label=annos[i], **kwargs)
        #n, bins_, patches = plt.hist(data[i], bins=bins[i], label=annos[i], histtype='stepfilled', alpha=0.3, normed=True, density=True)
        #n, bins_, patches = plt.hist(data[i], bins='auto', label=annos[i], histtype='stepfilled', alpha=0.3, normed=True, density=True)
    plt.xlabel(x_label)
    # plt.ylabel('Probability')
    # plt.title('Histogram of Cluster Size')
    # plt.xlim(0, bins.max())
    # plt.ylim(0, n.max())
    plt.grid(True)
    plt.legend()
    # plt.show()
    plt.savefig(hist_fn)
    plt.close(fig)



# def plot_hist(data, bins, work_dir=None):
def cluster_size_hist(data, bins, annos, work_dir=None, x_label='Cluster Size'):
    if work_dir is not None:
        if not os.path.isdir(work_dir):
            os.mkdir(work_dir)

    # Plot multiple saperate histograms
    for i in range(len(annos)):
        plot_single_hist(data[i], bins[i], annos[i], work_dir, x_label)
    # Plot all histograms in one figure
    if len(annos) > 1: 
        plot_multiple_hist(data, bins, annos, work_dir, x_label)



def cluster_data_processing(data, min_sz=None, max_sz=None):
    ids, counts = np.unique(data, return_counts=True)

    # Remove singletons
    if min_sz is not None:
        idx = counts >= min_sz
        counts = counts[idx]
    # Filter too large clusters
    if max_sz is not None:
        idx = counts <= max_sz
        counts = counts[idx]

    bins = np.unique(counts)
    # plot_hist(counts, bins, work_dir)

    return counts, bins

# TODO: Add a function to show the histogram of clusters within a certain range of pred_scores
def quality_score_processing(scores):
    print("To Implement ...")
