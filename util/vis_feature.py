#coding:utf-8
import sys
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def draw(feat_arr, label_arr):
    feat_embed = sklearn.decomposition.PCA(n_components=2).fit_transform(feat_arr)
    plt.figure(figsize=(10, 10))
    color_list = ["C"+str(each) for each in label_arr]
    plt.scatter(feat_embed[:, 0], feat_embed[:, 1], s=30, c=color_list)
    title = "show_dist"
    plt.title(title, fontsize=35, y=-0.1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    #plt.axis('off')
    outfile = "show_dist"
    plt.savefig(outfile)

if __name__ == "__main__":
    featfile, labelfile = sys.argv[1], sys.argv[2]
    feat_arr = np.load(featfile)
    label_arr = np.load(labelfile)
    draw(feat_arr, label_arr)
