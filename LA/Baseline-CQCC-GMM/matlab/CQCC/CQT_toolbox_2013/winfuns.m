function g = winfuns(name,x,L)
%WINFUNS  Window function generator  
%   Usage:  g = winfuns(name,x)
%           g = winfuns(name,N,L)
%           g = winfuns(name,N)
%   
%   Input parameters: 
%         name      : String containing the window name
%         x         : Vector of sampling positions
%         N         : Window support (in samples)
%         L         : Output length (in samples)
%   Output parameters:
%         g         : Output window
%   
%   This function serves to compute a variety of standard and some more 
%   exotic window functions. Most of the functions used are detailed and 
%   discussed in classical papers (see references below), but several are
%   included for special purposes in the toolbox only.
%   
%   Given a character string name containing the name of the desired
%   window function, the function offers 2 modes of operation. If the 
%   second input parameter is a vector x of sampling values, then the
%   specified function is evaluated at the given points. If a window length
%   N and optionally a signal length L are supplied, a symmetric, 
%   whole-point centered window with a support of N samples is produced 
%   and, given L, zero-extended to length L.
%
%   The following windows are available:
%
%     'hann'         von Hann window. Forms a PU. The Hann window has a
%                    mainlobe with of 8/N, a PSL of -31.5 dB and decay rate
%                    of 18 dB/Octave.
%
%     'cos'          Cosine window. This is the square root of the Hanning
%                    window. The cosine window has a mainlobe width of 6/N,
%                    a  PSL of -22.3 dB and decay rate of 12 dB/Octave.
%                  
%     'rec'          Rectangular window. The rectangular window has a
%                    mainlobe width of 4/N, a  PSL of -13.3 dB and decay
%                    rate of 6 dB/Octave. Forms a PU. Alias: 'square'
%
%     'tri'          Triangular window. 
%
%     'hamming'      Hamming window. Forms a PU that sums to 1.08 instead
%                    of 1.0 as usual. The Hamming window has a
%                    mainlobe width of 8/N, a  PSL of -42.7 dB and decay
%                    rate of 6 dB/Octave.
%
%     'blackman'     Blackman window. The Blackman window has a
%                    mainlobe width of 12/N, a PSL of -58.1 dB and decay
%                    rate of 18 dB/Octave.
%
%     'blackharr'    Blackman-Harris window. The Blackman-Harris window has 
%                    a mainlobe width of 16/N, a PSL of -92.04 dB and decay
%                    rate of 6 dB/Octave.
%
%     'modblackharr'  Modified Blackman-Harris window. This slightly 
%                     modified version of the Blackman-Harris window has 
%                     a mainlobe width of 16/N, a PSL of -90.24 dB and decay
%                     rate of 18 dB/Octave.
%
%     'nuttall'      Nuttall window. The Nuttall window has a mainlobe 
%                    width of 16/N, a PSL of -93.32 dB and decay rate of 
%                    18 dB/Octave.
%
%     'nuttall10'    2-term Nuttall window with 1 continuous derivative. 
%                    Alias: 'hann'.
%
%     'nuttall01'    2-term Nuttall window with 0 continuous derivatives. 
%                    Alias: 'hamming'.
%
%     'nuttall20'    3-term Nuttall window with 3 continuous derivatives. 
%                    The window has a mainlobe width of 12/N, a PSL of 
%                    -46.74 dB and decay rate of 30 dB/Octave.
%
%     'nuttall11'    3-term Nuttall window with 1 continuous derivative. 
%                    The window has a mainlobe width of 12/N, a PSL of 
%                    -64.19 dB and decay rate of 18 dB/Octave.
%
%     'nuttall02'    3-term Nuttall window with 0 continuous derivatives. 
%                    The window has a mainlobe width of 12/N, a PSL of 
%                    -71.48 dB and decay rate of 6 dB/Octave.
%
%     'nuttall30'    4-term Nuttall window with 5 continuous derivatives. 
%                    The window has a mainlobe width of 16/N, a PSL of 
%                    -60.95 dB and decay rate of 42 dB/Octave.
%
%     'nuttall21'    4-term Nuttall window with 3 continuous derivatives. 
%                    The window has a mainlobe width of 16/N, a PSL of 
%                    -82.60 dB and decay rate of 30 dB/Octave.
%
%     'nuttall12'    4-term Nuttall window with 1 continuous derivatives. 
%                    Alias: 'nuttall'.
%
%     'nuttall03'    4-term Nuttall window with 0 continuous derivatives. 
%                    The window has a mainlobe width of 16/N, a PSL of 
%                    -98.17 dB and decay rate of 6 dB/Octave.
%
%     'gauss'        Truncated, stretched Gaussian: exp(-18*x^2) restricted
%                    to the interval ]-.5,.5[.
%
%     'wp2inp'       Warped Wavelet uncertainty equalizer (see WP 2 of the
%                    EU funded project UnlocX). This function is included 
%                    as a test function for the Wavelet transform 
%                    implementation and serves no other purpose in this 
%                    toolbox.
%
%   See also:  nsgcqwin, nsgwvltwin, nsgerbwin
%
%   References:
%     Wikipedia. Window function - wikipedia article.
%     http://en.wikipedia.org/wiki/Window_function.
%     
%     A. Nuttall. Some windows with very good sidelobe behavior. IEEE Trans.
%     Acoust. Speech Signal Process., 29(1):84-91, 1981.
%     
%     F. Harris. On the use of windows for harmonic analysis with the
%     discrete Fourier transform. Proceedings of the IEEE, 66(1):51 - 83,
%     January 1978.
%     
%
%   Url: http://nsg.sourceforge.net/doc/windows/winfuns.php

% Copyright (C) 2013 Nicki Holighaus.
% This file is part of NSGToolbox version 0.1.0
% 
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 3.0 Unported 
% License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/3.0/ 
% or send a letter to 
% Creative Commons, 444 Castro Street, Suite 900, 
% Mountain View, California, 94041, USA.

% Author: Nicki Holighaus
% Date: 25.04.13

if nargin < 2
    error('Not enough input arguments');
end

if numel(x) == 1
    N = x;
    if nargin < 3
        L = N;
    end
    if L<N
        error('Output length L must be larger than or equal to N');
    end
    if mod(N,2) == 0 % For even N the sampling interval is [-.5,.5-1/N]
        x = [0:1/N:.5-1/N,-N*ones(1,L-N),-.5:1/N:-1/N]';
    else % For odd N the sampling interval is [-.5+1/(2N),.5-1/(2N)]
        x = [0:1/N:.5-.5/N,-N*ones(1,L-N),-.5+.5/N:1/N:-1/N]';
    end
end

if size(x,2) > 1
    x = x.';
end

switch name    
    case {'Hann','hann','nuttall10','Nuttall10'}
        g = .5 + .5*cos(2*pi*x);
        
    case {'Cosine','cosine','cos','Cos','sqrthann','Sqrthann'}
        g = cos(pi*x);
        
    case {'hamming','nuttall01','Hamming','Nuttall01'}
        g = .54 + .46*cos(2*pi*x);
        
    case {'square','rec','Square','Rec'}
        g = double(abs(x) < .5);
        
    case {'tri','triangular','bartlett','Tri','Triangular','Bartlett'}
        g = 1-2*abs(x);
        
    case {'blackman','Blackman'}
        g = .42 + .5*cos(2*pi*x) + .08*cos(4*pi*x);
        
    case {'blackharr','Blackharr'}
        g = .35875 + .48829*cos(2*pi*x) + .14128*cos(4*pi*x) + ...
            .01168*cos(6*pi*x);
        
    case {'modblackharr','Modblackharr'}
        g = .35872 + .48832*cos(2*pi*x) + .14128*cos(4*pi*x) + ...
            .01168*cos(6*pi*x);
        
    case {'nuttall','nuttall12','Nuttall','Nuttall12'}
        g = .355768 + .487396*cos(2*pi*x) + .144232*cos(4*pi*x) + ...
            .012604*cos(6*pi*x);
        
    case {'nuttall20','Nuttall20'}
        g = 3/8 + 4/8*cos(2*pi*x) + 1/8*cos(4*pi*x);
        
    case {'nuttall11','Nuttall11'}
        g = .40897 + .5*cos(2*pi*x) + .09103*cos(4*pi*x);
        
    case {'nuttall02','Nuttall02'}
        g = .4243801 + .4973406*cos(2*pi*x) + .0782793*cos(4*pi*x);
        
    case {'nuttall30','Nuttall30'}
        g = 10/32 + 15/32*cos(2*pi*x) + 6/32*cos(4*pi*x) + ...
            1/32*cos(6*pi*x);
        
    case {'nuttall21','Nuttall21'}
        g = .338946 + .481973*cos(2*pi*x) + .161054*cos(4*pi*x) + ...
            .018027*cos(6*pi*x);
        
    case {'nuttall03','Nuttall03'}
        g = .3635819 + .4891775*cos(2*pi*x) + .1365995*cos(4*pi*x) + ...
            .0106411*cos(6*pi*x);
        
    case {'gauss','truncgauss','Gauss','Truncgauss'}
        g = exp(-18*x.^2);
        
    case {'wp2inp','Wp2inp'}
        g = exp(exp(-2*x)*25.*(1+2*x));
        g = g/max(g);
        
    otherwise
        error('Unknown window function: %s.',name);
end;

% Force the window to 0 outside (-.5,.5)
g = g.*(abs(x) < .5);    
