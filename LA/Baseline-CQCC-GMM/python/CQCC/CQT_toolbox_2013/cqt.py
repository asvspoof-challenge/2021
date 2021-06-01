import numpy as np
import math
from CQCC.CQT_toolbox_2013.nsgcqwin import nsgcqwin
from CQCC.CQT_toolbox_2013.nsgtf_real import nsgtf_real
from CQCC.CQT_toolbox_2013.cqtCell2Sparse import cell2mat, cqtCell2Sparse

def cqt(*args):
    # %CQT  Constant-Q/Variable-Q transform
    # %   Usage:  Xcq = cqt(x, B, fs, fmin, fmax, varargin)
    # %
    # %   Input parameters:
    # %         x         : input signal
    # %         B         : number of bins per octave
    # %         fs        : sampling frequency
    # %         fmin      : lowest frequency to be analyzed
    # %         fmax      : highest frequency to be analyzed
    # %         varargin  : Optional input pairs (see table below)
    # %
    # %   Output parameters: 
    # %         Xcq       : Struct consisting of 
    # %           .c           : CQT coefficients
    # %           .cDC         : transform coefficients for f = 0
    # %           .cNyq        : transform coefficients for fs/2
    # %           .g           : cell array of analysis filters
    # %           .shift       : center frequencies of analysis filters
    # %           .M           : bandwidth of analysis filters
    # %           .xlen        : length of input signal
    # %           .phasemode   : 'local'  -> zero-centered filtered used
    # %                        : 'global' -> mapping function used
    # %           .rast        : time-frequency plane sampling scheme (full,
    # %                          piecewise, none)
    # %           .fmin
    # %           .fmax
    # %           .B       
    # %           .format      : eighter 'cell' or 'matrix' (only applies for
    # %                          piecewise rasterization)
    # %   
    # %   Optional input arguments arguments can be supplied like this:
    # %
    # %       Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'piecewise')
    # %
    # %   The arguments must be character strings followed by an
    # %   argument:
    # %
    # %     'rasterize':  can be set to (default is 'full');
    # %           - 'none':      Hop sizes are distinct for each frequency
    # %                          channel. Transform coefficients will be
    # %                          presented in a cell array.
    # %           - 'full':      The hop sizes for all freqency channels are 
    # %                          set to the smallest hop size in the representa-
    # %                          tion. Transform coefficients will be presented 
    # %                          in matrix format.
    # %           - 'piecewise': Hop sizes will be rounded down to be a power-of-
    # %                          two integer multiple of the smallest hop size in
    # %                          the representation. Coefficients will be 
    # %                          presented either in a sparse matrix or as cell 
    # %                          arrays (see 'format' option)
    # %
    # %     'phasemode':  can be set to (default is 'global')
    # %           - 'local':     Zero-centered filtered used
    # %           - 'global':    Mapping function used (see reference)
    # %
    # %     'format':     applies only for piecewise rasterization               
    # %           - 'sparse':   Coefficients will be presented in a sparse matrix 
    # %           - 'cell':     Coefficients will be presented in a cell array
    # %
    # %     'gamma':      the bandwidth of each filter is given by
    # %                            Bk = 1/Q * fk + gamma,
    # %                   where fk is the filters center frequency, Q is fully
    # %                   determined by the number of bins per octave and gamma
    # %                   is a bandwidth offset. If gamma = 0 the obtained
    # %                   filterbank is constant-Q. Setting gamma > 0 time
    # %                   resolution towards lower frequencies can be improved
    # %                   compared to the constant-Q case (e.g. ERB proportional
    # %                   bandwidths). See reference for more information.
    # %     'normalize':  coefficient normalization
    # %          - 'sine':    Filters are scaled such that a sinusoid with
    # %                       amplitude A in time domain will exhibit the same
    # %                       amplitude in the time-frequency representation.
    # %          - 'impulse': Filters are scaled such that an impulse in time
    # %                       domain will exhibit a flat response in the
    # %                       time-frequency representation (in the frame that 
    # %                       centers the impulse)
    # %          - 'none':      ...
    # %     'winfun':        defines the window function that is used for filter
    # %                   design. See winfuns for more information.
    # %
    # %   See also:  nsgtf_real, winfuns
    # %
    # %   References:
    # %     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�rfler. A Matlab 
    # %     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
    # %     Transforms.
    # %
    # %     G. A. Velasco, N. Holighaus, M. D�rfler, and T. Grill. Constructing an
    # %     invertible constant-Q transform with non-stationary Gabor frames.
    # %     Proceedings of DAFX11, Paris, 2011.
    # %     
    # %     N. Holighaus, M. D�rfler, G. Velasco, and T. Grill. A framework for
    # %     invertible, real-time constant-q transforms. Audio, Speech, and
    # %     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
    # %     
    # %
    # %
    # % Copyright (C) 2013 Christian Sch�rkhuber.
    # % 
    # % This work is licensed under the Creative Commons 
    # % Attribution-NonCommercial-ShareAlike 3.0 Unported 
    # % License. To view a copy of this license, visit 
    # % http://creativecommons.org/licenses/by-nc-sa/3.0/ 
    # % or send a letter to 
    # % Creative Commons, 444 Castro Street, Suite 900, 
    # % Mountain View, California, 94041, USA.

    # % Authors: Christian Sch�rkhuber
    # % Date: 20.09.13

    # check input arguments
    nargin = len(args)
    #defaults
    rasterize = 'full' # fully rasterized
    phasemode = 'global'
    outputFormat = 'sparse' # only applies if rasterize == 'octave'
    normalize = 'sine'
    windowFct = 'hann'
    gamma = 0

    x, B, fs, fmin, fmax = args[:5]

    if nargin >= 6:
        varargin = args[5:]
        Larg = len(varargin)
        # print(varargin)
        for ii in range(0,Larg,2):
            if varargin[ii] == 'rasterize':
                rasterize = varargin[ii+1]
            elif varargin[ii] == 'phasemode':
                phasemode = varargin[ii+1]
            elif varargin[ii] == 'format':
                outputFormat = varargin[ii+1]
            elif varargin[ii] == 'gamma':
                gamma = varargin[ii+1]
            elif varargin[ii] == 'normalize':
                normalize = varargin[ii+1]
            elif varargin[ii] == 'win':
                windowFct = varargin[ii+1]

    # print("cqt_fmin:", fmin)
    # print("cqt_fmax:", fmax)
    # print("cqt_B", B)
    # print("cqt_fs", fs)
    # print("cqt_len(x)", len(x))
    # print("cqt_windowFct", windowFct)
    # print("cqt_gamma", gamma)
    # window design
    g,shift,M = nsgcqwin(fmin, fmax, B, fs, len(x), 'winfun', windowFct, 'gamma', gamma, 'fractional', 0)
    
    

    
    fbas = fs*np.cumsum(shift[1:]) / len(x)
    
    # print(fbas)


    fbas = fbas[:int(M.shape[0]/2)-1]
    # print(fbas.shape)
    # print(fbas)
    

    # compute coefficients
    bins = int(M.shape[0]/2) - 1
    # print(bins)
    # print(rasterize)
    # print(M)

    if rasterize == 'full':
        # print(M)
        # print(max(M))
        # print(min(M))
        M[1:bins+1] = M[bins]
        # print(max(M))
        # print(M.shape)
        M[bins+2:] = M[bins:0:-1]
        # print(M.shape)
        # print(max(M))
        # print(M)
    elif rasterize == 'piecewise':
        temp = M[bins+1-1]
        octs = math.ceil(math.log(fmax/fmin, 2))
        # %make sure that the number of coefficients in the highest octave is
        # %dividable by 2 at least octs-times
        temp = math.ceil(temp/2**octs)*2**octs      
        mtemp = temp / M
        mtemp = 2 ** ( math.ceil(math.log(mtemp, 2)) -1)
        mtemp = temp / mtemp
        mtemp[bins+2-1] = M[bins+2-1] # don't rasterize Nyquist bin
        mtemp[1-1] = M[1-1] # don't rasterize DC bin
        M = mtemp

    # print(normalize)
    if normalize in {'sine','Sine','SINE','sin'}:
        normFacVec = 2*M[:bins+2] / len(x)
    elif normalize in {'impulse','Impulse', 'IMPULSE','imp'}:
        normFacVec = 2*M[:bins+2-1] / [len(cell) for cell in g]
    elif normalize in {'none','None','NONE','no'}:
        normFacVec = np.ones((bins+2,1))
    else:
        raise VauleError('Unkown normalization method!')
    
    # print(normFacVec)
    # print(normFacVec.shape)
    normFacVec = np.append(normFacVec, normFacVec[-2:0:-1])

    # print(max(normFacVec))
    # print(normFacVec.shape)

    g = g[:(2*bins+2)] * normFacVec[:(2*bins+2)]
    g = g.T

    # print(g[-1])

    # print(shift)
    # print(shift.shape)
    # print(max(shift))
    # print(phasemode)

    c, _ = nsgtf_real(x, g, shift, M, phasemode)    # note that returned c is a list

    # print(len(c))

    if rasterize == 'full': 
        
        # print(c[0].shape)
        # print(c[0][:5])
        cDC = cell2mat(c[0])
        # print(cDC.shape)  
        # print(cDC[:,:5]) 
        # print(bins)
        # print(c[bins+2-1][:5])

        cNyq = cell2mat(c[bins+2-1])  
        # print(cNyq.shape)
        # print(cNyq[:,:5])
        # print(c)
        # print(bins)
        c = cell2mat(c[1:bins+1])
        # print(c.shape)
        # print(c[:,:5])

    elif rasterize == 'piecewise':
        cDC = cell2mat(c[0])   
        cNyq = cell2mat(c[bins+2-1])
        if outputFormat == 'sparse':
            c = cqtCell2Sparse(c,M).T
        else:
            c = c[1:-1]
    else:
        cDC = cell2mat(c[0])   
        cNyq = cell2mat(c[-1])
        c = c[1:-1]

    # output   
    Xcq = {'c': c.T, 'g': g, 'shift': shift, 'M': [M], 
        'xlen': len(x), 'phasemode': phasemode, 'rast': rasterize, 
        'fmin': fmin, 'fmax': fmax, 'B': B, 'cDC': cDC, 'cNyq': cNyq, 
        'format': outputFormat, 'fbas': fbas}

    return Xcq