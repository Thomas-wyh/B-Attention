# B-Attention

This is an official implementation for "Robust Graph Structure Learning over Images via Multiple Statistical Tests" accepted at NeurIPS 2022.

## News

- :fire: **B-Attenion** is accepted by NeurIPS 2022!

- :fire: The previous method [**Ada-NETS**](https://github.com/damo-cv/Ada-NETS) is accepted by ICLR 2022!
  
  ## Introduction
  
  Graph structure learning aims to learn connectivity in a graph from data. It is particularly important for many computer vision related tasks since no explicit graph structure is available for images for most cases. A natural way to construct a graph among images is to treat each image as a node and assign pairwise image similarities as weights to corresponding edges. It is well known that pairwise similarities between images are sensitive to the noise in feature representations, leading to unreliable graph structures. We address this problem from the viewpoint of statistical tests. By viewing the feature vector of each node as an independent sample, the decision of whether creating an edge between two nodes based on their similarity in feature representation can be thought as a *single* statistical test. To improve the robustness in the decision of creating an edge, multiple samples are drawn and integrated by *multiple* statistical tests to generate a more reliable similarity measure, consequentially more reliable graph structure. The corresponding elegant matrix form named **B**-Attention is designed for efficiency. The effectiveness of multiple tests for graph structure learning is verified both theoretically and empirically on multiple clustering and ReID benchmark datasets.
  
  <img src=image/b-attention.png width=600 height=350 />

  ## Main Results on Feature Quality
  <img src=image/feature_quality.png width=900 height=320 />

  ## Main Results on Face clustering for Downstream Tasks
  <img src=image/experiment_results.png width=900 height=355 />

  ## Case studies on self attention, **Q**-Attention and **B**-Attention
  <img src=image/case_studies.png width=900 height=1050 />
  
 
  
  ## Getting Started
  ### Install
+ Clone this repo
  
  ```
  git clone https://github.com/Thomas-wyh/Ada-NETS
  cd Ada-NETS
  ```

+ Create a conda virtual environment and activate it
  
  ```
  conda create -n adanets python=3.6 -y
  conda activate adanets
  ```

+ Install `Pytorch` , `cudatoolkit` and other requirements.
  
  ```
  conda install pytorch==1.2 torchvision==0.4.0a0 cudatoolkit=10.2 -c pytorch
  pip install -r requirements.txt
  ```
- Install `Apex`:
  
  ```
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  ```
  
  ### Data preparation
  
  The process of clustering on the MS-Celeb part1 is as follows:
  The original data files are from [here](https://github.com/yl-1993/learn-to-cluster/blob/master/DATASET.md#supported-datasets)(The feature and label files of MSMT17 used in Ada-NETS are [here](http://idstcv.oss-cn-zhangjiakou.aliyuncs.com/Ada-NETS/MSMT17/msmt17_feature_label.zip)). For convenience, we convert them to `.npy` format after L2 normalized. The original features' dimension is 256. The file structure should look like:
  
  ```
  data
  ├── feature
  │   ├── part0_train.npy
  │   └── part1_test.npy
  └── label
    ├── part0_train.npy
    └── part1_test.npy
  ```
  
  Build the $k$NN by faiss:
  
  ```
  sh script/faiss_search.sh
  ```
  
  Obtain the top$K$ neighbours and distances of each vertex in the structure space:
  
  ```
  sh script/struct_space.sh
  ```
  
  Obtain the best neigbours by the candidate neighbours quality criterion:
  
  ```
  sh script/max_Q_ind.sh
  ```
  
  ### Train with the B-Attention
    
  Train with the B-Attention:
  
  ```
  bash script/train.sh
  ```
  
  And then get the enhanced vertex features:
  
  ```
  bash script/get_feat.sh  
  ```
  
  Perform feature quality evaluation, and the mAP, rank1 index will be printed and ROC curve will be generated.

  ``
  bash script/eval_feat_quality.sh
  ``
  
  Perform cluster faces on MS-Celeb-1M part1:
  
  ```
  bash script/get_cluster.sh
  ```
  
  It will print the evaluation results of clustering.
  
  ## Acknowledgement
  
  This code is based on the publicly available face clustering [learn-to-cluster](https://github.com/yl-1993/learn-to-cluster), [CDP](https://github.com/XiaohangZhan/cdp), [BallClustering](https://github.com/makarandtapaswi/BallClustering_ICCV2019), [Ada-NETS](https://github.com/damo-cv/Ada-NETS) and the [dmlc/dgl](https://github.com/dmlc/dgl).
  The k-nearest neighbor search tool uses [faiss](https://github.com/facebookresearch/faiss).
  
  ## Citing B-Attention
  
  ```
  @inproceedings{
  wang2022robust,
  title={Robust Graph Structure Learning over Images via Multiple Statistical Tests},
  author={Yaohua Wang and Fangyi Zhang and Ming Lin and Senzhang Wang and Xiuyu Sun and Rong Jin},
  booktitle={Thirty-Sixth Conference on Neural Information Processing Systems},
  year={2022},
  url={https://openreview.net/forum?id=VVCI8-PYYv}
  }
  ```
