function [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls,varargin)
%NSGCQWIN  Constant-Q/Variable-Q dictionary generator
%   Usage:  [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls,varargin)
%           [g,shift,M] = nsgcqwin(fmin,fmax,bins,sr,Ls)
%
%   Input parameters:
%         fmin      : Minimum frequency (in Hz)
%         fmax      : Maximum frequency (in Hz)
%         bins      : number of bins per octave
%         sr        : Sampling rate (in Hz)
%         Ls        : Length of signal (in samples)
%         varargin  : Optional input pairs (see table below)
%   Output parameters: 
%         g         : Cell array of constant-Q/variable-Q filters
%         shift     : Vector of shifts between the center frequencies
%         M         : Vector of lengths of the window functions
%
%   Create a nonstationary Gabor filterbank with constant or varying 
%   Q-factor and relevant frequency range from fmin to fmax. To allow
%   for perfect reconstruction, the frequencies outside that range will be
%   captured by 2 additional filters placed on the zero and Nyquist
%   frequencies, respectively.
%
%   The Q-factor (quality factor) is the ratio of center frequency to
%   bandwidth cent_freq/bandwidth.
%
%
%   For more details on the construction of the constant-Q nonstationary 
%   Gabor filterbank, please check the reference.
%   
%   Optional input arguments arguments can be supplied like this:
%
%       nsgcqwin(fmin,fmax,bins,sr,Ls,'min_win',min_win)
%
%   The arguments must be character strings followed by an
%   argument:
%
%     'min_win',min_win  Minimum admissible window length (in samples) 
%
%     'bwfac',bwfac            Channel numbers M are rounded to multiples 
%                              of this
%
%     'fractional',fractional  Allow fractional shifts and bandwidths
%
%     'winfun',winfun          String containing the desired window 
%                              function name
%
%     'gamma':      the bandwidth of each filter is given by
%                            Bk = 1/Q * fk + gamma,
%                   where fk is the filters center frequency, Q is fully
%                   determined by the number of bins per octave and gamma
%                   is a bandwidth offset. If gamma = 0 the obtained
%                   filterbank is constant-Q. Setting gamma > 0 time
%                   resolution towards lower frequencies can be improved
%                   compared to the constant-Q case (e.g. ERB proportional
%                   bandwidths). See reference for more information.
%
%   See also: nsgtf_real, winfuns
%
%   References:
%     C. Schörkhuber, A. Klapuri, N. Holighaus, and M. Dörfler. A Matlab 
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
%     Transforms.
%
%     G. A. Velasco, N. Holighaus, M. DÃ¶rfler, and T. Grill. Constructing an
%     invertible constant-Q transform with non-stationary Gabor frames.
%     Proceedings of DAFX11, Paris, 2011.
%     
%     N. Holighaus, M. DÃ¶rfler, G. Velasco, and T. Grill. A framework for
%     invertible, real-time constant-q transforms. Audio, Speech, and
%     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
%     
%
%   Url: http://nsg.sourceforge.net/doc/generators/nsgcqwin.php

% Copyright (C) 2013 Nicki Holighaus, Christian Schörkhuber.
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 3.0 Unported 
% License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/3.0/ 
% or send a letter to 
% Creative Commons, 444 Castro Street, Suite 900, 
% Mountain View, California, 94041, USA.

% Authors: Nicki Holighaus, Gino Velasco, Monika Doerfler
% Date: 25.04.13
% Edited by Christian Schörkhuber, 25.09.2013

% Set defaults
bwfac = 1;
min_win = 4;
fractional = 0;
winfun = 'hann';
gamma = 0;

% Check input arguments

if nargin < 5
    error('Not enough input arguments');
end

if nargin >= 6
    Lvar = length(varargin);
    if mod(Lvar,2)
        error('Invalid input argument');
    end
    for kk = 1:2:Lvar
        if ~ischar(varargin{kk})
            error('Invalid input argument');
        end
        switch varargin{kk}
            case {'min_win'}
                min_win = varargin{kk+1};
            case {'gamma'}
                gamma = varargin{kk+1};
            case {'bwfac'}
                bwfac = varargin{kk+1};
            case {'fractional'}
                fractional = varargin{kk+1};
            case {'winfun'}
                winfun = varargin{kk+1};
            otherwise
                error(['Invalid input argument: ', varargin{kk}]);
        end
    end
end

nf = sr/2;

if fmax > nf
    fmax = nf;
end

fftres = sr / Ls;
b = floor(bins * log2(fmax/fmin));
fbas = fmin .* 2.^((0:b).'./bins);

Q = 2^(1/bins) - 2^(-1/bins);
cqtbw = Q*fbas + gamma; cqtbw = cqtbw(:);

%make sure the support of highest filter won't exceed nf
tmpIdx = find(fbas+cqtbw/2>nf,1,'first');
if (~isempty(tmpIdx))
    fbas = fbas(1:tmpIdx-1);
    cqtbw = cqtbw(1:tmpIdx-1);
end

%make sure the support of the lowest filter won't exceed DC
tmpIdx = find(fbas-cqtbw/2<0,1,'last');
if (~isempty(tmpIdx))
    fbas = fbas(tmpIdx+1:end);
    cqtbw = cqtbw(tmpIdx+1:end);
    warning(['fmin set to' num2str(fftres*floor(fbas(1)/fftres),6)  ' Hz!']);
end

Lfbas = length(fbas);
fbas = [0;fbas];
fbas(Lfbas+2) = nf;
fbas(Lfbas+3:2*(Lfbas+1)) = sr-fbas(Lfbas+1:-1:2);

bw = [2*fmin; cqtbw; fbas(Lfbas+3)-fbas(Lfbas+1); cqtbw(end:-1:1)];
fftres = sr / Ls;
bw = bw / fftres;
fbas = fbas / fftres;

% center positions of filters in DFT frame
posit = zeros(size(fbas));
posit(1:Lfbas+2) = floor(fbas(1:Lfbas+2));
posit(Lfbas+3:end) = ceil(fbas(Lfbas+3:end));

shift = [mod(-posit(end),Ls); diff(posit)];

if fractional
    corr_shift = fbas-posit;
    M = ceil(bw+1);
else
    bw = round(bw);
    M = bw;
end

for ii = 1:2*(Lfbas+1)
    if bw(ii) < min_win;
        bw(ii) = min_win;
        M(ii) = bw(ii);
    end
end

if fractional
    g = arrayfun(@(x,y,z) ...
        winfuns(winfun,([0:ceil(z/2),-floor(z/2):-1]'-x)/y)/sqrt(y),corr_shift,...
        bw,M,'UniformOutput',0);
else
        g = arrayfun(@(x) winfuns(winfun,x),...
            bw,'UniformOutput',0);

end


M = bwfac*ceil(M/bwfac);

% Setup Tukey window for 0- and Nyquist-frequency
for kk = [1,Lfbas+2]
    if M(kk) > M(kk+1);
        g{kk} = ones(M(kk),1);
        g{kk}((floor(M(kk)/2)-floor(M(kk+1)/2)+1):(floor(M(kk)/2)+...
            ceil(M(kk+1)/2))) = winfuns('hann',M(kk+1));
        g{kk} = g{kk}/sqrt(M(kk));
    end
end
