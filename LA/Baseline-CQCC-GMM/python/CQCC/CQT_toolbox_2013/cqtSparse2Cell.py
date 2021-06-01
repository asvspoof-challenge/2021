import numpy as np

'''
1. cell
2. full
'''

def cqtSparse2Cell(cSparse,M,cDC,cNyq):

    bins = M.shape[0]/2 - 1
    # cell
    cCell = cell(1,bins+2)
    cCell{bins+2} = cNyq

    M = M[0:bins+1]
    step = 1
    # full
    cSparse = full(cSparse)
    distinctHops = np.log2(M[bins]/M[1])+1
    curNumCoef = M[bins]


    for ii in range(distinctHops):
        idx = (M == curNumCoef)
        # cSparse
        temp = cSparse(idx,1:step:end).T
        temp = num2cell(temp,1)
        cCell(idx) = temp
        step = step * 2
        curNumCoef = curNumCoef / 2

    cCell{1} = cDC

    return cCell
