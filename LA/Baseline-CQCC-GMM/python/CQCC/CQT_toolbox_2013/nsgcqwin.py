import numpy as np
import math
import warnings 
from CQCC.CQT_toolbox_2013.winfuns import winfuns

# function [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls,varargin)
def nsgcqwin(*args):
    # %NSGCQWIN  Constant-Q/Variable-Q dictionary generator
    # %   Usage:  [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls,varargin)
    # %           [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls)
    # %
    # %   Input parameters:
    # %         fmin      : Minimum frequency (in Hz)
    # %         fmax      : Maximum frequency (in Hz)
    # %         bins      : number of bins per octave
    # %         sr        : Sampling rate (in Hz)
    # %         Ls        : Length of signal (in samples)
    # %         varargin  : Optional input pairs (see table below)
    # %   Output parameters: 
    # %         g         : Cell array of constant-Q/variable-Q filters
    # %         shift     : Vector of shifts between the center frequencies
    # %         M         : Vector of lengths of the window functions
    # %
    # %   Create a nonstationary Gabor filterbank with constant or varying 
    # %   Q-factor and relevant frequency range from fmin to fmax. To allow
    # %   for perfect reconstruction, the frequencies outside that range will be
    # %   captured by 2 additional filters placed on the zero and Nyquist
    # %   frequencies, respectively.
    # %
    # %   The Q-factor (quality factor) is the ratio of center frequency to
    # %   bandwidth cent_freq/bandwidth.
    # %
    # %
    # %   For more details on the construction of the constant-Q nonstationary 
    # %   Gabor filterbank, please check the reference.
    # %   
    # %   Optional input arguments arguments can be supplied like this:
    # %
    # %       nsgcqwin(fmin,fmax,bins,sr,Ls,'min_win',min_win)
    # %
    # %   The arguments must be character strings followed by an
    # %   argument:
    # %
    # %     'min_win',min_win  Minimum admissible window length (in samples) 
    # %
    # %     'bwfac',bwfac            Channel numbers M are rounded to multiples 
    # %                              of this
    # %
    # %     'fractional',fractional  Allow fractional shifts and bandwidths
    # %
    # %     'winfun',winfun          String containing the desired window 
    # %                              function name
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
    # %
    # %   See also: nsgtf_real, winfuns
    # %
    # %   References:
    # %     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�rfler. A Matlab 
    # %     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
    # %     Transforms.
    # %
    # %     G. A. Velasco, N. Holighaus, M. Dörfler, and T. Grill. Constructing an
    # %     invertible constant-Q transform with non-stationary Gabor frames.
    # %     Proceedings of DAFX11, Paris, 2011.
    # %     
    # %     N. Holighaus, M. Dörfler, G. Velasco, and T. Grill. A framework for
    # %     invertible, real-time constant-q transforms. Audio, Speech, and
    # %     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
    # %     
    # %
    # %   Url: http://nsg.sourceforge.net/doc/generators/nsgcqwin.php

    # % Copyright (C) 2013 Nicki Holighaus, Christian Sch�rkhuber.
    # % This work is licensed under the Creative Commons 
    # % Attribution-NonCommercial-ShareAlike 3.0 Unported 
    # % License. To view a copy of this license, visit 
    # % http://creativecommons.org/licenses/by-nc-sa/3.0/ 
    # % or send a letter to 
    # % Creative Commons, 444 Castro Street, Suite 900, 
    # % Mountain View, California, 94041, USA.

    # % Authors: Nicki Holighaus, Gino Velasco, Monika Doerfler
    # % Date: 25.04.13
    # % Edited by Christian Sch�rkhuber, 25.09.2013

    # Set defaults
    bwfac = 1
    min_win = 4
    fractional = 0
    winfun = 'hann'
    gamma = 0
    nargin = len(args)

    # Check input arguments
    if nargin < 5:
        raise VauleError('Not enough input arguments.')

    if nargin >= 6:
        fmin, fmax, bins, sr, Ls = args[:5]
        varargin = args[5:]
        Lvar = len(varargin)
        assert Lvar % 2 == 0, 'Invalid input argument'
        for kk in range(0, Lvar, 2):
            if not isinstance(varargin[kk], str):
                raise ValueError('Invalid input argument.')
            if varargin[kk] == 'min_win':
                min_win = varargin[kk+1]
            elif varargin[kk] == 'gamma':
                gamma = varargin[kk+1]
            elif varargin[kk] == 'bwfac':
                bwfac = varargin[kk+1]
            elif varargin[kk] == 'fractional':
                fractional = varargin[kk+1]
            elif varargin[kk] == 'winfun':
                winfun = varargin[kk+1]
            else: 
                raise TypeError('Invalid input argument: ', varargin[kk])

    nf = sr/2

    # print("nsgcqwin_gamma:", gamma)
    # print("nsgcqwin_bwfac:", bwfac)
    # print("nsgcqwin_fractional:", fractional)
    # print("nsgcqwin_winfun:", winfun)

    if fmax > nf:
        fmax = nf

    fftres = sr / Ls
    b = math.floor(bins * math.log(fmax/fmin, 2))

    fbas = fmin * 2 ** (np.array(list(range(b+1))).T / bins)

    Q = 2**(1/bins) - 2**(-1/bins)
    cqtbw = Q*fbas + gamma
    cqtbw = cqtbw[:]
    # print(fmax)
    # print(fmin)
    # print(fftres)
    # print(b)
    # # print(bins)
    # print(fbas.shape)
    # # print(fbas)
    # print(Q)
    # print(cqtbw.shape)
    # print(cqtbw)

    # make sure the support of highest filter won't exceed nf
    tmpIdx = np.where(fbas+cqtbw/2 > nf)[0]
    if not np.all(tmpIdx == 0):
        fbas = fbas[:tmpIdx[0]]
        cqtbw = cqtbw[:tmpIdx[0]]

    # print(tmpIdx)
    # print(fbas)
    # print(cqtbw)

    # make sure the support of the lowest filter won't exceed DC
    tmpIdx = np.where(fbas-cqtbw/2<0)[0]
    if not np.all(tmpIdx == 0):
        fbas = fbas[tmpIdx[-1]+1:]
        cqtbw = cqtbw[tmpIdx[-1]+1:]
        warnings.warn('fmin set to' + str(fftres*math.floor(fbas[0]/fftres),6) + ' Hz!')

    # print(tmpIdx)
    # print(fbas)
    # print(cqtbw)

    Lfbas = len(fbas)
    # print(Lfbas)

    # print(max(fbas))

    fbas = np.insert(fbas, 0, 0)
    # print("fbas max", max(fbas))
    fbas = np.insert(fbas, len(fbas), nf)
    fbas = np.insert(fbas, len(fbas), sr-fbas[Lfbas:0:-1])
    # print("fbas max", max(fbas))
    # print(fbas)
    
    bw = cqtbw[::-1] 
    # print(bw)
    # print(max(bw))
    # print(Lfbas+3)
    # print("fbas", fbas[Lfbas+3-1])
    # print(fbas[Lfbas+1-1])
    bw = np.insert(bw, 0, fbas[Lfbas+3-1]-fbas[Lfbas+1-1])
    # print(max(bw))

    bw = np.insert(bw, 0, cqtbw)
    bw = np.insert(bw, 0, 2*fmin)
    # print(max(bw))
    # print(len(bw))

    fftres = sr / Ls
    # print(bw)
    bw = bw / fftres
    # print(bw)
    fbas = fbas / fftres

    # print(fftres)
    # print(bw)
    # print(fbas)

    # center positions of filters in DFT frame
    # print(fbas.shape)

    # print(fbas)
    posit = np.zeros(fbas.shape)
    # print(fbas)
    posit[:Lfbas+2] = np.array([math.floor(fbas_num) for fbas_num in list(fbas[:Lfbas+2])])
    # print(len(posit))
    posit[Lfbas+2:] = np.array([math.ceil(fbas_num) for fbas_num in list(fbas[Lfbas+2:])])
    # print(posit)
    # print(len(posit))
    
    # posit [0, 62, 63, ..., 64181, 64181, 64182]
    # py_posit [0, 62, 63, ..., 64180, 64181, 64181]

    # print(posit)
    # print(len(posit))
    shift = np.diff(posit)
    # print(shift)
    shift = np.insert(shift, 0, -posit[-1] % Ls)
    # print(shift)
    # print(len(shift))

    # print(bw)

    if fractional:
        corr_shift = fbas-posit
        M = math.ceil(bw+1)
    else:
        bw = np.round(bw)
        M = bw


    for ii in range(2*(Lfbas+1)):
        if bw[ii] < min_win:
            bw[ii] = min_win
            M[ii] = bw[ii]
    
    # print(max(M))
    # print(M[200:400])
    # print(bw)
    # print(M)
    # print(len(M))

    # print(fractional)

    if fractional:
        ary = (np.array(list(range(math.ceil(M/2)+1)))+np.array(list(range(-math.floor(M/2),0)))).conj().T
        g = winfuns(winfun, (ary-corr_shift) / bw) / math.sqrt(y)
    else:
        # print(winfun)
        ##### TO DO #####
        g = np.empty_like(bw, dtype=object)   # cell array 
        for i in range(len(bw)):
            g[i] = winfuns(winfun, bw[i])
        # print(g)

    # print(g.shape)
    # print(g)
    # print(bwfac)
    # print(bwfac)
    # print(max(M))
    M = bwfac*np.ceil(M/bwfac)
    # print(M[100])
    # Setup Tukey window for 0- and Nyquist-frequency
    # print(Lfbas)
    for kk in range(Lfbas+2):
        if M[kk] > M[kk+1]:
            g[kk] = np.ones((int(M[kk]),1))
            start = int((np.floor(M[kk]/2)-np.floor(M[kk+1]/2)+1))
            # print(start)
            end = int((np.floor(M[kk]/2)+np.ceil(M[kk+1]/2)))
            # print(end)
            g[kk][start-1:end] = winfuns('hann', M[kk+1])
            g[kk] = g[kk]/np.sqrt(M[kk])
    
    # print(g)   # cell array 
    # print(shift)  # [62, 62, 1, ... , 1, 0, 1]
    # print(M)
    # print(max(M))

    return g, shift, M
