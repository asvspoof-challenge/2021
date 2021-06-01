import math
import numpy as np

def winfuns(*args):

    nargin = len(args)

    if nargin < 2:
        raise ValueError('Not enough input arguments')
    
    name = args[0]
    x = args[1]

    if x.size == 1:
        
        N = x
        # print(N)
        if nargin < 3:
            L = N
        # print(L)

        if L < N:
            raise ValueError('Output length L must be larger than or equal to N')
        
        if N % 2 == 0:

            x1 = np.linspace(0, 0.5-1/N, int((0.5-1/N)*N)+1)
            x1 = x1.reshape(1, x1.shape[0])

            if L > N:
                x2 = -N*np.ones((1,L-N))
            else:
                x2 = np.array([]).reshape(1,0)

            x3 = np.linspace(-0.5, -1/N, int((0.5-1/N)*N)+1)
            x3 = x3.reshape(1, x3.shape[0])

            x = np.concatenate((x1, x2, x3), axis=-1).conj().T
            # print(x.shape)
            # print(x)
        else:
            x1 = np.linspace(0, 0.5-0.5/N, int((0.5-0.5/N)*N)+1)
            # print(x1)
            x1 = x1.reshape(1, x1.shape[0])
            # print(x1)
            if L > N:
                x2 = -N*np.ones((1,L-N))
            else:
                x2 = np.array([]).reshape(1,0)
            # print(x2)
            x3 = np.linspace(-0.5+0.5/N, -1/N, int((0.5-0.5/N)*N-1)+1)
            x3 = x3.reshape(1, x3.shape[0])
            # print(x3)
            # print(x1.shape)
            # print(x2.shape)
            # print(x3.shape)
            x = np.concatenate((x1, x2, x3), axis=-1).conj().T
            # print(x)
            # print(x.shape)

    # print(x.shape)

    if x.shape[1] > 1:
        x = x.T
    
    # print(name)
    if name in {'Hann','hann','nuttall10','Nuttall10'}:
        g = 0.5 + 0.5*np.cos(2*np.pi*x)
        # print(g.shape)
        # print(g)
    elif name in {'Cosine','cosine','cos','Cos','sqrthann','Sqrthann'}:
        g = np.cos(np.pi*x)
    elif name in {'hamming','nuttall01','Hamming','Nuttall01'}:
        g = 0.54 + 0.46*np.cos(2*np.pi*x)
    elif name in {'square','rec','Square','Rec'}:
        g = (abs(x) < 0.5).double()
    elif name in {'tri','triangular','bartlett','Tri','Triangular','Bartlett'}:
        g = 1 - 2*abs(x)
    elif name in {'blackman','Blackman'}:
        g = 0.42 + 0.5*np.cos(2*np.pi*x) + 0.08*np.cos(4*np.pi*x)
    elif name in {'blackharr','Blackharr'}:
        g = 0.35875 + 0.48829*np.cos(2*np.pi*x) + 0.14128*np.cos(4*np.pi*x) + 0.01168*np.cos(6*np.pi*x)
    elif name in {'modblackharr','Modblackharr'}:
        g = 0.35872 + 0.48832*np.cos(2*np.pi*x) + 0.14128*np.cos(4*np.pi*x) + 0.01168*np.cos(6*np.pi*x)
    elif name in {'nuttall','nuttall12','Nuttall','Nuttall12'}:
        g = 0.355768 + 0.487396*np.cos(2*np.pi*x) + 0.144232*np.cos(4*np.pi*x) + 0.012604*np.cos(6*np.pi*x)
    elif name in {'nuttall20','Nuttall20'}:
        g = 3/8 + 4/8*np.cos(2*np.pi*x) + 1/8*np.cos(4*np.pi*x)
    elif name in {'nuttall11','Nuttall11'}:
        g = 0.40897 + 0.5*np.cos(2*np.pi*x) + 0.09103*np.cos(4*np.pi*x)
    elif name in {'nuttall02','Nuttall02'}:
        g = 0.4243801 + 0.4973406*np.cos(2*np.pi*x) + 0.0782793*np.cos(4*np.pi*x)
    elif name in {'nuttall30','Nuttall30'}:
        g = 10/32 + 15/32*np.cos(2*np.pi*x) + 6/32*np.cos(4*np.pi*x) + 0.1/32*np.cos(6*np.pi*x)
    elif name in {'nuttall21','Nuttall21'}:
        g = 0.338946 + 0.481973*np.cos(2*np.pi*x) + 0.161054*np.cos(4*np.pi*x) + 0.018027*np.cos(6*np.pi*x)
    elif name in {'nuttall03','Nuttall03'}:
        g = 0.3635819 + 0.4891775*np.cos(2*np.pi*x) + 0.1365995*np.cos(4*np.pi*x) + 0.0106411*np.cos(6*np.pi*x)
    elif name in {'gauss','truncgauss','Gauss','Truncgauss'}:
        g = np.exp(-18*x**2)
    elif name in {'wp2inp','Wp2inp'}:
        g = np.exp(np.exp(-2*x)*25*(1+2*x))
        g = g / max(g)
    else:
        raise ValueError('Unknown window function: %s.',name)
        
    g = g * (abs(x) < 0.5)

    # print(g)

    return g
