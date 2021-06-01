import numpy as np
import math


def cqtFillSparse(c, M, B):

# %Repeat coefficients in sparse matrix until the next valid coefficient. 
# %For visualization this is an overkill since we could image each CQT bin
# %seperately, however, in some case this might come in handy.

    bins = c.shape[1]
    M = M[:bins]
    distinctHops = math.log(M[bins]/M[2-1], 2)+1

    curNumCoef = M[-1-1] / 2
    step = 2
    for ii in range(distinctHops -1):  
        idx = [M == curNumCoef]
        idx += list(range(0, len(c), step))
        temp = c[idx]
        temp = np.tile(temp, (step, 1))
        temp = np.reshape(temp[:], ((idx!=0).sum(), []))
        idx += list(range(len(c)))
        c[idx] = temp
        step = 2*step
        curNumCoef = curNumCoef / 2

    return c
