#coding:utf-8
import sys
import numpy as np
import os
from multiprocessing import Pool

def get_sim(pair):
    p1, p2 = pair
    sim = np.dot(feat[p1], feat[p2])
    return sim

if __name__ == "__main__":
    featfile, posfile, negfile, outpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    feat = np.load(featfile)

    pospair_list = np.load(posfile)
    negpair_list = np.load(negfile)

    pool = Pool(50)
    res_list = pool.map(get_sim, pospair_list)
    pool.close()
    pool.join()
    np.save(os.path.join(outpath, 'pos_res'), res_list)
    print("pos done")

    pool = Pool(50)
    res_list = pool.map(get_sim, negpair_list)
    pool.close()
    pool.join()
    np.save(os.path.join(outpath, 'neg_res'), res_list)
    print("neg done")

