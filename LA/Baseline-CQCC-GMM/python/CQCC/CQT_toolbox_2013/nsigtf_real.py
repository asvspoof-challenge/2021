import numpy as np

def nsigtf_real(*args):

    nargin = len(args)

    if nargin < 4:
        raiseValueError('Not enough input arguments')
    
    # iscell
    if iscell(c) == 0:
        if np.ndims(c) == 2:
            N, chan_len = c.shape
            CH = 1
            # 
            c = mat2cell(c.',chan_len,ones(1,N)).')
        else:
            N, chan_len, CH = c.shape
            ctemp = mat2cell(np.permute(c,[2,1,3]),chan_len,np.ones((1,N)),np.ones((1,CH)))
            c = np.permute(ctemp,[2,3,1])
    else:
        CH, N = c.shape

    posit = np.cumsum(shift)
    NN = posit[-1]
    posit = posit - shift[1]

    fr = np.zeros((NN,CH))

    for ii in range(N):
        Lg = len(g[ii])
        win_range = (posit[ii]+(-np.floor(Lg/2):math.ceil(Lg/2)-1)) % NN + 1
        temp = np.fft(c[ii], [], 1) * len(c[ii])

        if phasemode == 'global':
            fsNewBins = c[ii].shape[0]
            fkBins = posit[ii]
            displace = fkBins - np.floor(fkBins/fsNewBins) * fsNewBins
            temp = np.roll(temp, -displace)
        
        # 
        temp = temp[]

        fr[win_range,:] = fr[win_Range,:] + temp * g[ii][Lg-np.floor(Lg/2)+1:Lg, 1:np.ceil(Lg/2)]
    
        nyqBin = np.floor(Ls/2) + 1

        #
        fr[nyqBin+1:end] = conj( fr(nyqBin  - (~logical(mod(Ls,2))) : -1 : 2) )

        fr = np.real(np.fft.ifft(fr))

    return fr