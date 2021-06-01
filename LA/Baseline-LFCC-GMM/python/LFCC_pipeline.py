from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.fbanks.linear_fbanks import linear_filter_banks
from spafe.utils.cepstral import cms, cmvn, lifter_ceps
from spafe.utils.spectral import dct, power_spectrum
from librosa import util
import numpy as np


def lfcc(sig,
         fs=16000,
         num_ceps=20,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.030,
         win_hop=0.015,
         win_type="hamming",
         nfilts=70,
         nfft=1024,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         normalize=0):
    """
    Compute the linear-frequency cepstral coefﬁcients (GFCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of LFCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)
    
    # -> FFT -> |.|
    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform)**2

    #  -> x linear-fbanks
    linear_fbanks_mat = linear_filter_banks(nfilts=nfilts,
                                            nfft=nfft,
                                            fs=fs,
                                            low_freq=low_freq,
                                            high_freq=high_freq,
                                            scale=scale)

    features = np.dot(abs_fft_values, linear_fbanks_mat.T)
    
    log_features = np.log10(features+2.2204e-16)

    #  -> DCT(.)
    lfccs=dct(log_features, type=dct_type, norm='ortho', axis=1)[:, :num_ceps ]

    return lfccs
