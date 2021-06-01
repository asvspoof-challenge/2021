import numpy as np

'''
1. firpmord
2. firpm
'''

def hp(fs):

    Fstop = 0.125
    Fpass = 0.25
    Dstop = 0.001
    Dpass = 0.057501127785
    dens  = 20

    # firpmord
    N, Fo, Ao, W = firpmord([Fstop, Fpass]/(fs/2), [0 1], [Dstop, Dpass])

    # firpm
    b  = firpm(N, Fo, Ao, W, {dens})

    return b
