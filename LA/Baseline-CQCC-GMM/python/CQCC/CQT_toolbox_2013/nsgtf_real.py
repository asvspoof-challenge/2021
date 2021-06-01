import numpy as np
import math 


''' TO DO
1. fft 

'''

def nsgtf_real(*args):

    # %NSGTF_REAL  Nonstationary Gabor filterbank for real signals
    # %   Usage: [c,Ls] = nsgtf_real(f,g,shift,M, phasemode)
    # %          c = nsgtf_real(...)
    # %
    # %   Input parameters: 
    # %         f         : A real-valued signal to be analyzed (For multichannel
    # %                     signals, input should be a matrix which each
    # %                     column storing a channel of the signal).
    # %         g         : Cell array of analysis filters
    # %         shift     : Vector of frequency shifts
    # %         M         : Number of time channels (optional).
    # %                     If M is constant, the output is converted to a
    # %                     matrix
    # %         phasemode : 'local': zero-centered filtered used
    # %                     'global': mapping function used (see cqt)
    # %   Output parameters:
    # %         c         : Transform coefficients (matrix or cell array)
    # %         Ls        : Original signal length (in samples)
    # %
    # %   Given the cell array g of windows, the time shift vector shift, and
    # %   channel numbers M, NSGTF_REAL computes the corresponding 
    # %   nonstationary Gabor filterbank of f, using only the filters with at 
    # %   least partially supported on the positive frequencies. Let 
    # %   P(n)=sum_{l=1}^{n} shift(l), then the output 
    # %   c = NSGTF_REAL(f,g,shift,M) is a cell array with 
    # %
    # %              Ls-1                                      
    # %      c{n}(m)= sum fft(f)(l)*conj(g\{n\}(l-P(n)))*exp(2*pi*i*(l-P(n))*m/M(n))
    # %               l=0                                      
    # %
    # %   where m runs from 0 to M(n)-1 and n from 1 to N, where
    # %   g{N} is the final filter at least partially supported on the
    # %   positive frequencies. All filters in g, shift that are completely
    # %   supported on the negative frequencies are ignored.
    # %
    # %   For more details, see NSGTF.
    # %
    # %   See also:  nsigtf_real, nsdual, nstight
    # %

    # % Copyright (C) 2013 Nicki Holighaus.
    # % This file is part of NSGToolbox version 0.1.0
    # % 
    # % This work is licensed under the Creative Commons 
    # % Attribution-NonCommercial-ShareAlike 3.0 Unported 
    # % License. To view a copy of this license, visit 
    # % http://creativecommons.org/licenses/by-nc-sa/3.0/ 
    # % or send a letter to 
    # % Creative Commons, 444 Castro Street, Suite 900, 
    # % Mountain View, California, 94041, USA.

    # % Author: Nicki Holighaus, Gino Velasco
    # % Date: 23.04.13
    # % Edited by Christian Sch�rkhuber, 25.09.2013

    # Check input arguments
    nargin = len(args)
    if nargin < 2:
        raise ValueError('Not enough input arguments.')

    f = args[0] 
    g = args[1]

    # print(f.shape)

    Ls, CH = f.shape

    # print(Ls, CH)

    if Ls == 1:
        f = f.T
        Ls = CH
        CH = 1

    if CH > Ls:
        print('The number of signal channels (' + str(CH) + ') is larger than')
        print('the number of samples per channel (' + str(Ls) + ').')
        reply = input('Is this correct? ([Y]es,[N]o)')
        if reply in {'N','n','No','no',''}:
            reply2 = input('Transpose signal matrix? ([Y]es,[N]o)')
            if reply2 in {'N','n','No','no',''}:
                    raise ValueError('Invalid signal input, terminating program')
            elif reply2 in {'Y','y','Yes','yes'}:
                print('Transposing signal matrix and continuing program execution')
                f = f.T
                X = CH
                CH = Ls
                Ls = CH
                del X
            else:
                raise ValueError('Invalid reply, terminating program') 
        elif reply in {'Y','y','Yes','yes'}:
            print('Continuing program execution')
        else:
            raise ValueError('Invalid reply, terminating program')

    shift = args[2] 
    M = args[3]
    phasemode = args[4]

    N = len(shift)    # The number of frequency slices
    # print(N)

    if nargin == 3:
        M = np.zeros((N,1))
        for kk in range(N):
            M[kk] = len(g[kk])

    if max(M.shape) == 1:
        M = M[0]*np.ones((N,1))


    # some preparation
    # print(f)
    # print(f.shape)
    # print(max(f))
    # import scipy
    # from pyfftw.interfaces import scipy_fftpack as fftw

    # print(f)
    # print(f.shape)

    # f = np.squeeze(f, axis=-1)
    # import numpy.matlib as M
    f = np.fft.fft(f, axis=0)     ###### TO DO fft Python 
    # matlab: (447 + 348j)  max
    # python: (491 + 182j)  max

    # print(f)
    # print(f.shape)
    # print(max(f))

    posit = np.cumsum(shift)-shift[0] # Calculate positions from shift vector
    # print(posit)
    # print(posit.shape)
    # print(posit[:5])

    # A small amount of zero-padding might be needed (e.g. for scale frames)
    fill = int(np.sum(shift)-Ls)
    # print(fill)
    # print(np.zeros((int(fill), CH)).shape)
    f = np.concatenate([f, np.zeros((fill, CH))], axis=0)
    # print(f.shape)
    # print(f)

    Lg = np.array([len(cell) for cell in g]) # cellfun(@length,g);
    # print(Lg)
    # print(posit-np.floor(Lg/2))
    # print((Ls+fill)/2)

    N = np.where(posit-np.floor(Lg/2) <= (Ls+fill)/2)[0][-1] 
    # print(N)   # 864

    # c = np.empty((N+1,1),dtype=object)  # Initialisation of the result
    # print(c.shape)   # (865, 1)
    # print(c)
    c = []

    ########## TO DO ########## ########## ########## ########## ########## ########## ########## 
    
    # The actual transform
    for ii in range(N+1):

        # print("ii:", ii)

        idx1 = np.arange((np.ceil(Lg[ii]/2)+1-1), Lg[ii])
        idx = np.append(idx1, np.arange(0, np.ceil(Lg[ii]/2)))

        idx = np.array(idx, dtype=int)

        # print(idx.shape)
        # print(posit[ii])

        # print((posit[ii]-np.floor(Lg[ii]/2)))
        # print(np.ceil(Lg[ii]/2))
        # print()

        win_range = (posit[ii] + np.arange(int(-np.floor(Lg[ii]/2)), int(np.ceil(Lg[ii]/2)))) % (Ls+fill)
        win_range = np.array(win_range, dtype=int)
        # print(win_range.shape)
        # print(win_range)
        # print(M[ii])
        # print(Lg[ii])

        if M[ii] < Lg[ii]: # if the number of frequency channels is too small,
            # aliasing is introduced (non-painless case)
            col = np.ceil(Lg[ii]/M[ii])
            temp = np.zeros((col*M[ii],CH))
            end = col*M[ii]
            idx_list = list(range(end-np.floor(Lg[ii]/2)+1-1, end)) + list(range(0,np.ceil(Lg[ii]/2)))
            temp[idx_list,:] = f[win_range,:] * g[ii][idx]
            temp = np.reshape(temp,(M[ii],col,CH))
        
            c.append(np.squeeze(np.fft.ifft(np.sum(temp,axis=1))))
            # % Using c = cellfun(@(x) squeeze(ifft(x)),c,'UniformOutput',0);
            # % outside the loop instead does not provide speedup; instead it is
            # % slower in most cases.
        else:
            
            temp = np.zeros((int(M[ii]),CH), dtype=complex)
            # print(temp.shape)
            end = int(M[ii])
            # print(end)
            # print(end-np.floor(Lg[ii]/2)+1-1)

            idx_list = list(range(int(end-np.floor(Lg[ii]/2)+1-1), end)) + list(range(0,int(np.ceil(Lg[ii]/2))))
            # print(idx_list)
            idx_array = np.array(idx_list)
            # print(idx_array)
            # print(f.shape)
            # print(g[ii][idx].shape)
            # print(f.shape)
            # print(win_range)
            # print(f[win_range].shape)
            # print(ii)
            # print(idx)
            temp[idx_array,:] = f[win_range] * g[ii][idx]

            if phasemode == 'global':
                #apply frequency mapping function (see cqt)
                fsNewBins = int(M[ii])
                # print(fsNewBins)

                fkBins = int(posit[ii])
                # print(fkBins)

                displace = int(fkBins - np.floor(fkBins/fsNewBins) * fsNewBins)
                # temp = circshift(temp, displace)  

                # if ii == N:
                #     print(temp.shape)
                #     print(temp[:5])
                #     print(displace)
                # temp = np.squeeze(temp, axis=-1)
                # if ii == N:
                #     print(temp.shape)
                #     print(temp)
               
                # temp = temp.reshape((-1, 1))
                # if ii == N:
                #     a = temp[:5]
                #     print(a)
                #     a = circshift(a, 2)
                #     print(a)
                    # print(temp[:5])
                temp = circshift(temp, displace)
                # if ii == N:
                #     print(temp.shape)
                #     print(temp[:5])
            # print(temp.shape)
            # print(np.fft.ifft(temp))
            c.append(np.fft.ifft(temp, axis=0))
    #         c{ii} = c{ii}.* ( 2* M(ii)/Lg(ii) ); %energy normalization
    # print(c[-1].shape)
    # print(c[-1][:5])
    # print(len(c))

    if max(M) == min(M):

        c_list = [c[i][0] for i in range(len(c))]
        c = np.vstack(c_list).astype(c[0].dtype) # cell2mat(c)  
        c = np.reshape(c,(M[0],N,CH))

    return c, Ls

def circshift(temp, displace):
    temp = np.roll(temp, displace)
    # print(temp)
    # temp = np.roll(temp, -displace)
    # print(temp)
    return temp


