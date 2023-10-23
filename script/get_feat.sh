#!/bin/bash
set -euxo pipefail
set +x

path_prefix=<prefix>
train_featfile=$path_prefix/data/part0_train_feature.npy
train_orderadjfile=$path_prefix/data/train_adj_top256.npz
train_adjfile=$path_prefix/data/original/other/train_adj_top120.npz
train_labelfile=$path_prefix/data/part0_train_label.npy
test_featfile=$path_prefix/data/part1_test_feature.npy
test_adjfile=$path_prefix/data/original/other/test_adj_top120.npz
test_labelfile=$path_prefix/data/part1_test_label.npy

phase=test
losstype=allmax
margin=1.0
pmargin=0.95
pweight=10.0
topk=120
temperature=1.0
outpath=output
ckpt=ckpt_100000
model_path=$outpath/${ckpt}.pth

param="--config_file config.yml --outpath $outpath --phase $phase
    --train_featfile $train_featfile --train_adjfile $train_adjfile
    --train_labelfile $train_labelfile --train_orderadjfile $train_orderadjfile
    --test_featfile $test_featfile --test_adjfile $test_adjfile
    --test_labelfile $test_labelfile --losstype $losstype --topk $topk --temperature $temperature
    --resume_path $model_path"


time CUDA_VISIBLE_DEVICES=0 python -u train.py $param

bash get_search.sh
