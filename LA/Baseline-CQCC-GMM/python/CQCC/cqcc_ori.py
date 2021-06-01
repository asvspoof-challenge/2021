import librosa 
import numpy as np
import scipy


def cqt(sig, fs=16000, low_freq=10, high_freq=3000, b=48):
    """
    Compute the constant Q-transform.
          - take the absolute value of the FFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components
    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        low_freq  (int) : lowest band edge of mel filters (Hz).
                          Default is 10.
        high_freq (int) : highest band edge of mel filters (Hz).
                          Default is 3000.
        b         (int) : number of bins per octave.
                          Default is 48.
    Returns:
        array including the Q-transform coefficients.
    """

    # define lambda funcs for clarity
    def f(k):
        return low_freq * 2**((k - 1) / b)

    def w(N):
        return np.hamming(N)

    def nk(k):
        return np.ceil(Q * fs / f(k))

    def t(Nk, k):
        return (1 / Nk) * w(Nk) * np.exp(
            2 * np.pi * 1j * Q * np.arange(Nk) / Nk)

    # init vars
    Q = 1 / (2**(1 / b) - 1)
    K = int(np.ceil(b * np.log2(high_freq / low_freq)))
    print(K)
    nfft = int(2**np.ceil(np.log2(Q * fs / low_freq)))

    # define temporal kernal and sparse kernal variables
    S = [
        scipy.sparse.coo_matrix(np.fft.fft(t(nk(k), k), nfft))
        for k in range(K, 0, -1)
    ]
    S = scipy.sparse.vstack(S[::-1]).tocsc().transpose().conj() / nfft

    # compute the constant Q-transform
    xcq = (np.fft.fft(sig, nfft).reshape(1, nfft) * S)[0]
    return xcq
    

def cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD):

    
    # Step 1: cqt 
    # xcq = cqt(x, fs, fmin, fmax, B)

    # xcq = librosa.feature.chroma_cqt(x, fs, fmin=fmin, bins_per_octave=B)

    absCQt = abs(xcq)

    print(absCQt.shape)




    # Step 2: power spectrum 





    # Step 3: log power spectrum




    # Step 4: uniform resampling





    # Step 5: dct



    # Step 6: CQCC formula 

