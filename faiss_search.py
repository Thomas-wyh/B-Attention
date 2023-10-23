#coding:utf-8
import numpy as np
import faiss
from tqdm import tqdm
import sys
import time
import os
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz


def topk_to_edges_and_scores(nbrs, cos_dist):
    k = nbrs.shape[1]
    # query = nbrs[:,0]
    query = np.arange(nbrs.shape[0])
    queries = np.expand_dims(query, axis=1)
    queries = np.repeat(queries, k, axis=1)

    edges = np.concatenate((queries.reshape(-1,1), nbrs.reshape(-1,1)), axis=1)
    scores = 1 - cos_dist.reshape(-1)

    return edges, scores


def save_as_spmatrix(nbrs, cos_dist, prefix='.', symmetric=True, thr=0.0):
    n_samples = nbrs.shape[0]
    edges, scores = topk_to_edges_and_scores(nbrs, cos_dist)

    if thr > 0.0:
        inds = np.where(scores >= thr)[0]
        scores = scores[inds]
        edges = edges[inds,:]

    adj = csr_matrix((scores, (edges[:,0].tolist(), edges[:,1].tolist())), shape=(n_samples, n_samples))

    if symmetric:
        # binary_adj = adj > 0.0
        binary_adj = adj.copy()
        binary_adj[binary_adj.nonzero()] = 1.0
        binary_adj = binary_adj.astype(np.int)
        binary_adj = binary_adj + binary_adj.transpose()
        adj = adj + adj.transpose()
        adj[adj.nonzero()] = adj[adj.nonzero()] / binary_adj[binary_adj.nonzero()]

    file_path = os.path.join(prefix, 'adj_top%s.npz'%nbrs.shape[1])
    save_npz(file_path, adj)

    return adj


def batch_search(index, query, topk, bs, verbose=False):
    n = len(query)
    dists = np.zeros((n, topk), dtype=np.float32)
    nbrs = np.zeros((n, topk), dtype=np.int32)

    for sid in tqdm(range(0, n, bs), desc="faiss searching...", disable=not verbose):
        eid = min(n, sid + bs)
        dists[sid:eid], nbrs[sid:eid] = index.search(query[sid:eid], topk)
    cos_dist = dists / 2
    return cos_dist, nbrs


def search(query_arr, doc_arr, outpath, tag, topk, save_file=True):
    ### parameter
    nlist = 100  # 1000 cluster for 100w
    nprobe = 100    # test 10 cluster
    # topk = 1024
    bs = 100
    ### end parameter


    beg_time = time.time()
    #print("configure faiss")
    num_gpu = faiss.get_num_gpus()
    dim = query_arr.shape[1]
    #cpu_index = faiss.index_factory(dim, 'IVF100', faiss.METRIC_INNER_PRODUCT)
    quantizer = faiss.IndexFlatL2(dim)
    cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    cpu_index.nprobe = nprobe

    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=num_gpu)

    # start IVF
    #print("build index")
    gpu_index.train(doc_arr)
    gpu_index.add(doc_arr)
    #print(gpu_index.ntotal)

    # start query
    #print("start query")
    gpu_index.nprobe = nprobe # default nprobe is 1, try a few more
    D, I = batch_search(gpu_index, query_arr, topk, bs, verbose=True)

    if tag == 'sp':
        save_as_spmatrix(I, D, prefix=outpath)
    elif save_file:
        np.save(os.path.join(outpath, tag+'D'), D)
        np.save(os.path.join(outpath, tag+'I'), I)
        #data = np.concatenate((I[:,None,:], D[:,None,:]), axis=1)
        #np.savez(os.path.join(outpath,'data'), data=data)
    #print("time use", time.time()-beg_time)
    return I, D

if __name__ == "__main__":
    #print(sys.argv)
    if len(sys.argv) == 5:
        queryfile, docfile, outpath, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
        topk = 1024
    else:
        queryfile, docfile, outpath, tag, topk = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    #print("load feat file")
    query_arr = np.load(queryfile)
    doc_arr = np.load(docfile)

    search(query_arr, doc_arr, outpath, tag, int(topk))
