#!/bin/bash
set -euxo pipefail
set +x

outpath=output
origIfile=$outpath/fcI.npy
localDfile=$outpath/fcD.npy
simfile=$outpath/fcsim
time python get_mat2dict.py $origIfile $localDfile $simfile

path_prefix=<prefix>
test_labelfile=$path_prefix/data/part1_test_label.npy
python -u eval_feature_quality.py $origIfile $localDfile $simfile $test_labelfile $outpath
