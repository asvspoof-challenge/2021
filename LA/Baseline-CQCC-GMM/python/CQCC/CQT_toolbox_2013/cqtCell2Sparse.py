import numpy as np
import math
import scipy.sparse as sps

def cqtCell2Sparse(c, M):

    bins = M.shape[1]/2 - 1
    spLen = M[bins+1-1]
    cSparse = np.zeros((bins,spLen))

    M = M[:bins+1-1]

    step = 1

    distinctHops = math.log(M[bins+1-1]/M[2], 2)+1
    curNumCoef = M[bins+1-1]

    for ii in range(distinctHops):
        idx = [M == curNumCoef] + [false]

        temp = cell2mat(c[idx].T).T
        idx += list(range(0,len(cSparse), step))
        cSparse[idx] = temp
        step = step*2
        curNumCoef = curNumCoef / 2

    cSparse = sparse(cSparse)    # sparse return (index), value

    return cSparse


def cell2mat(c):
    # print("c.length:", len(c))
    c = np.stack(c)
    if c.ndim == 3:
        c = np.squeeze(c, axis=-1)
    c = c.T # cell2mat(c)
    return c

def sparse(m):
    index_res = np.where(m>0)
    index_list = [index for index in index_res]
    value_list = [m[index[0]][index[1]] for index in index_res]
    return index_list, value_list