#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import time
import sys
import os
import torch
from bertviz import model_view

def show_model_view(attention, tokens, hide_delimiter_attn=False, display_mode="dark"):
    if hide_delimiter_attn:
        for i, t in enumerate(tokens):
            if t in ("[SEP]", "[CLS]"):
                for layer_attn in attention:
                    layer_attn[0, :, i, :] = 0
                    layer_attn[0, :, :, i] = 0
    model_view(attention, tokens, display_mode=display_mode)

def visualize(feat_path, ind_path, label_path, attention_path):
    feats = np.load(feat_path)
    inds = np.load(ind_path)
    labels = np.load(label_path)
    attention = torch.load(attention_path)

    tokens = list()
    for i in range(len(inds)):
        tokens.append("%s_%s"%(labels[i],inds[i]))

    show_model_view(attention, tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize attention maps')
    # parser.add_argument('--prefix_path', type=str)
    parser.add_argument('--feat_path', type=str)
    parser.add_argument('--ind_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--attention_path', type=str)
    args = parser.parse_args()

    # prefix = args.prefix_path
    feat_path = args.feat_path
    ind_path = args.ind_path
    label_path = args.label_path
    attention_path = args.attention_path


    visualize(feat_path, ind_path, label_path, attention_path)