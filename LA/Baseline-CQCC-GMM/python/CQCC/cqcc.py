import numpy as np
import math
from CQCC.CQT_toolbox_2013.cqt import cqt
# import scikits.samplerate as sk_samplerate
import scipy
import librosa


def cqcc(*args):
    # %   Constant Q cepstral coefficients
    # %   Usage:  CQcc = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
    # %
    # %   Input parameters:
    # %         x        : input signal
    # %         fs       : sampling frequency
    # %         B        : number of bins per octave [default = 96]
    # %         fmax     : highest frequency to be analyzed [default = Nyquist frequency]
    # %         fmin     : lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
    # %         d        : number of uniform samples in the first octave [default 16]
    # %         cf       : number of cepstral coefficients excluding 0'th coefficient [default 19]
    # %         ZsdD     : any sensible combination of the following  [default ZsdD]:
    # %                      'Z'  include 0'th order cepstral coefficient
    # %                      's'  include static coefficients (c)
    # %                      'd'  include delta coefficients (dc/dt)
    # %                      'D'  include delta-delta coefficients (d^2c/dt^2)
    # %
    # %   Output parameters:
    # %         CQcc              : constant Q cepstral coefficients (nCoeff x nFea)
    # %         LogP_absCQT       : log power magnitude spectrum of constant Q trasform
    # %         TimeVec           : time at the centre of each frame [sec]
    # %         FreqVec           : center frequencies of analysis filters [Hz]
    # %         Ures_LogP_absCQT  : uniform resampling of LogP_absCQT
    # %         Ures_FreqVec      : uniform resampling of FreqVec [Hz]
    # %
    # %   See also:  cqt
    # %
    # %
    # %   References:
    # %     M. Todisco, H. Delgado, and N. Evans. A New Feature for Automatic
    # %     Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients.
    # %     Proceedings of ODYSSEY - The Speaker and Language Recognition
    # %     Workshop, 2016.
    # %
    # %     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�fler. A Matlab
    # %     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
    # %     Transforms. Proceedings AES 53rd Conference on Semantic Audio, London,
    # %     UK, Jan. 2014. http://www.cs.tut.fi/sgn/arg/CQT/
    # %
    # %     G. A. Velasco, N. Holighaus, M. D�fler, and T. Grill. Constructing an
    # %     invertible constant-Q transform with non-stationary Gabor frames.
    # %     Proceedings of DAFX11, Paris, 2011.
    # %
    # %     N. Holighaus, M. D�fler, G. Velasco, and T. Grill. A framework for
    # %     invertible, real-time constant-q transforms. Audio, Speech, and
    # %     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
    # %
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %
    # % Copyright (C) 2016 EURECOM, France.
    # %
    # % This work is licensed under the Creative Commons
    # % Attribution-NonCommercial-ShareAlike 4.0 International
    # % License. To view a copy of this license, visit
    # % http://creativecommons.org/licenses/by-nc-sa/4.0/
    # % or send a letter to
    # % Creative Commons, 444 Castro Street, Suite 900,
    # % Mountain View, California, 94041, USA.
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %
    # % Authors: Massimiliano Todisco {todisco [at] eurecom [dot] fr}
    # %          Hector Delgado {delgado [at] eurecom [dot] fr}
    # %
    # % Version: 1.0
    # % Date: 22.01.16
    # %
    # % User are requested to cite the following paper in papers which report 
    # % results obtained with this software package.	
    # %
    # %     M. Todisco, H. Delgado, and N. Evans. A New Feature for Automatic
    # %     Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients.
    # %     Proceedings of ODYSSEY - The Speaker and Language Recognition
    # %     Workshop, 2016.
    # %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # CHECK INPUT PARAMETERS
    nargin = len(args)
    # print("cqcc_nargin:", nargin)
    if nargin < 2:
        raise ValueError('Not enough input arguments.')

    x, fs = args[:2]

    # DEFAULT INPUT PARAMETERS
    if nargin < 3:
        B = 96
    else: 
        B = args[2]

    if nargin < 4: 
        fmax = fs/2
    else:
        fmax = args[3]

    if nargin < 5:
        oct = math.ceil(math.log(fmax/20, 2))
        fmin = fmax/2**oct
    else:
        fmin = args[4]

    if nargin < 6:
        d = 16
    else:
        d = args[5]

    if nargin < 7:
        cf = 19
    else:
        cf = args[6]

    if nargin < 8:
        ZsdD = 'ZsdD'
    else:
        ZsdD = args[7]

    gamma = 228.7*(2**(1/B)-2**(-1/B))
    # print("cqcc_gamma", gamma)

    # print("X:", x.shape)

    # CQT COMPUTING
    Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)

    # LOG POWER SPECTRUM
    absCQT = abs(Xcq['c'])
    # print(absCQT.shape)
    # print(absCQT[0][:5])

    TimeVec = np.arange(1, absCQT.shape[1]+1).reshape(1,-1)
    # print(TimeVec.shape)
    # print(TimeVec)
    TimeVec = TimeVec*Xcq['xlen'] / absCQT.shape[1] / fs
    # print(TimeVec.shape)
    # print(TimeVec[:,:5])

    FreqVec = np.arange(0, absCQT.shape[0]).reshape(1,-1)
    # print(FreqVec.shape)
    # print(FreqVec[:5])

    FreqVec = fmin*(2**(FreqVec/B))
    # print(FreqVec.shape)             # (1, 863)
    # print(FreqVec[:,:20])   
    # print(FreqVec[:,-20:])   

    eps = 2.2204e-16
    LogP_absCQT = np.log(absCQT**2 + eps)     
    # print(LogP_absCQT.shape)        # (863, 470)
    # print(LogP_absCQT[0][:5])


    # UNIFORM RESAMPLING
    kl = B*math.log(1+1/d, 2)
    # LogP_absCQT = np.asfortranarray(LogP_absCQT.T)
    import samplerate
    # import resampy

    fs = 1/(fmin*(2**(kl/B)-1))  # 1.024
    p = 1
    q = 1
    # new_sr = fs * q/p   
    # print(fs)

    ############ Output value a little bit different from results used by Matlab resample ############
    Ures_LogP_absCQT = librosa.resample(LogP_absCQT.T, fs, 9.562).T
    Ures_FreqVec = None
    # print(Ures_LogP_absCQT.shape)   # (8059, 470)     
    # print(Ures_LogP_absCQT[:10,0])
    # [Ures_LogP_absCQT, Ures_FreqVec] = resample(LogP_absCQT, FreqVec,1/(fmin*(2^(kl/B)-1)),1,1,'spline');
    # resample(LogP_absCQT, FreqVec,fs,1,1,'spline');

    # DCT
    CQcepstrum = scipy.fftpack.dct(Ures_LogP_absCQT, type=2, axis=1, norm='ortho')
    # print(CQcepstrum.shape)
    # print(CQcepstrum[:10,0])

    # DYNAMIC COEFFICIENTS
    if 'Z' in ZsdD:
        scoeff = 1
    else: 
        scoeff = 2

    CQcepstrum_temp = CQcepstrum[scoeff-1:cf+1,:]
    # print(CQcepstrum_temp.shape)

    f_d = 3 # delta window size

    if ZsdD.replace('Z','') == 'sdD':
        # print(Deltas(CQcepstrum_temp,f_d).shape)

        CQcc = np.concatenate([CQcepstrum_temp, Deltas(CQcepstrum_temp,f_d),
            Deltas(Deltas(CQcepstrum_temp,f_d),f_d)], axis=0)
        # print(CQcc.shape)

    elif ZsdD.replace('Z','') == 'sd':
        CQcc = np.concatenate([CQcepstrum_temp, Deltas(CQcepstrum_temp,f_d)], axis=0)

    elif ZsdD.replace('Z','') == 'sD':
        CQcc = np.concatenate([CQcepstrum_temp, Deltas(Deltas(CQcepstrum_temp,f_d),f_d)], axis=0)

    elif ZsdD.replace('Z','') == 's':
        CQcc = CQcepstrum_temp

    elif ZsdD.replace('Z','') == 'd':
        CQcc = Deltas(CQcepstrum_temp,f_d)

    elif ZsdD.replace('Z','') == 'D':
        CQcc = Deltas(Deltas(CQcepstrum_temp,f_d),f_d)

    elif ZsdD.replace('Z','') == 'dD':
        CQcc = np.concatenate([Deltas(CQcepstrum_temp,f_d), Deltas(Deltas(CQcepstrum_temp,f_d),f_d)], axis=0)

    return CQcc.T, LogP_absCQT.T, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT

def Deltas(x, hlen):
    # % Delta and acceleration coefficients
    # %
    # % Reference:
    # %   Young S.J., Evermann G., Gales M.J.F., Kershaw D., Liu X., Moore G., Odell J., Ollason D.,
    # %   Povey D., Valtchev V. and Woodland P., The HTK Book (for HTK Version 3.4) December 2006.
    # print(x.shape)
    # print(hlen)
    win = list(range(hlen, -hlen-1, -1))
    # print(win)
    
    xx_1 = np.tile(x[:,0],(1,hlen)).reshape(hlen,-1).T
    xx_2 = np.tile(x[:,-1],(1,hlen)).reshape(hlen,-1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    # print(xx.shape)     # (20, 476)

    #####filter function in Matlab#####
    D = scipy.signal.lfilter(win, 1, xx)  
    # print("after filter:", D.shape)
    
    D = D[:,hlen*2:]
    # print(D.shape)

    D = D /(2*sum(np.arange(1,hlen+1))**2)
    # print(D.shape)

    return D

