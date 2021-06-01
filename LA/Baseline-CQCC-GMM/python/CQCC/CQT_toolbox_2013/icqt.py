import numpy as np

'''
1. cell
2. num2cell
'''

def icqt(Xcq):

    Xcq.gd = {nsdual(Xcq.g,Xcq.shift,Xcq.M)}

    if Xcq.rast == 'piecewise':
        if Xcq.format == 'sparse':
            c = cqtSparse2Cell(Xcq.c,Xcq.M, Xcq.cDC, Xcq.cNyq)
        else:
            # cell
            c = cell(1,Xcq.shape[1]+2)
            c[1:-2] = Xcq.c[:]
            c[0] = {Xcq.cDC.T}
            c[-1] = {Xcq.cNyq}
    elif Xcq.rast == 'full':
        # cell
        c = cell(1,Xcq.shape[0]+2)
        # num2cell
        c[1:-2] = num2cell(Xcq.c,1)
        c[0] = {Xcq.cDC.T}
        c[-1] = {Xcq.cNyq.T}
    else:
        c = cell(1,Xcq.c.shape[1]+2)
        c[1:-2] = Xcq.c[:]
        c[0] = {Xcq.cDC}
        c[-1] = {Xcq.cNyq}

    x = nsigtf_real(c,Xcq.gd{1},Xcq.shift,Xcq.xlen, Xcq.phasemode)

    gd = Xcq.gd[0]

    return gd