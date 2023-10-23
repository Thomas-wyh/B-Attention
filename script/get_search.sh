#!/bin/bash
set -euxo pipefail
set +x

outpath=output
ckpt=ckpt_100000
featfile=${outpath}/fcfeat_${ckpt}.npy
tag=fc
time python -W ignore faiss_search.py $featfile $featfile $outpath $tag
# output fcI.npy fcD.npy
