#!/bin/bash
set -euxo pipefail
set +x

outpath=output
feat_file=$outpath/test_sample_sequence_feats.npy
ind_file=$outpath/test_sample_sequence_inds.npy
label_file=$outpath/test_sample_sequence_labels.npy
attention_file=$outpath/test_sample_attention_maps.pt

time python -u visualize_attention.py \
    --feat_path $feat_file \
    --ind_path $ind_file \
    --label_path $label_file \
    --attention_path $attention_file
