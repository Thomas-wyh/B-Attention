#coding:utf-8
import sys
import numpy as np

if __name__ == "__main__":
    Ifile, Dfile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]

    topN = 256
    topN = 121

    nbr_arr = np.load(Ifile).astype(np.int32)[:, :topN]
    dist_arr = np.load(Dfile)[:, :topN]

    out_dict = {}
    for query_nodeid, (nbr, dist) in enumerate(zip(nbr_arr, dist_arr)):
        for j, doc_nodeid in enumerate(nbr):
            if query_nodeid < doc_nodeid:
                tpl = (query_nodeid, doc_nodeid)
            else:
                tpl = (doc_nodeid, query_nodeid)
            out_dict[tpl] = 1 - dist[j]
        if query_nodeid % 1000 == 0:
            print("==>%s/%s"%(query_nodeid, len(nbr_arr)))
    np.save(outfile, out_dict)
