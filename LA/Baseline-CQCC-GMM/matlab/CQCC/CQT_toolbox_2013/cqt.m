function Xcq = cqt(x, B, fs, fmin, fmax, varargin)
%CQT  Constant-Q/Variable-Q transform
%   Usage:  Xcq = cqt(x, B, fs, fmin, fmax, varargin)
%
%   Input parameters:
%         x         : input signal
%         B         : number of bins per octave
%         fs        : sampling frequency
%         fmin      : lowest frequency to be analyzed
%         fmax      : highest frequency to be analyzed
%         varargin  : Optional input pairs (see table below)
%
%   Output parameters: 
%         Xcq       : Struct consisting of 
%           .c           : CQT coefficients
%           .cDC         : transform coefficients for f = 0
%           .cNyq        : transform coefficients for fs/2
%           .g           : cell array of analysis filters
%           .shift       : center frequencies of analysis filters
%           .M           : bandwidth of analysis filters
%           .xlen        : length of input signal
%           .phasemode   : 'local'  -> zero-centered filtered used
%                        : 'global' -> mapping function used
%           .rast        : time-frequency plane sampling scheme (full,
%                          piecewise, none)
%           .fmin
%           .fmax
%           .B       
%           .format      : eighter 'cell' or 'matrix' (only applies for
%                          piecewise rasterization)
%   
%   Optional input arguments arguments can be supplied like this:
%
%       Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'piecewise')
%
%   The arguments must be character strings followed by an
%   argument:
%
%     'rasterize':  can be set to (default is 'full');
%           - 'none':      Hop sizes are distinct for each frequency
%                          channel. Transform coefficients will be
%                          presented in a cell array.
%           - 'full':      The hop sizes for all freqency channels are 
%                          set to the smallest hop size in the representa-
%                          tion. Transform coefficients will be presented 
%                          in matrix format.
%           - 'piecewise': Hop sizes will be rounded down to be a power-of-
%                          two integer multiple of the smallest hop size in
%                          the representation. Coefficients will be 
%                          presented either in a sparse matrix or as cell 
%                          arrays (see 'format' option)
%
%     'phasemode':  can be set to (default is 'global')
%           - 'local':     Zero-centered filtered used
%           - 'global':    Mapping function used (see reference)
%
%     'format':     applies only for piecewise rasterization               
%           - 'sparse':   Coefficients will be presented in a sparse matrix 
%           - 'cell':     Coefficients will be presented in a cell array
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
%     'normalize':  coefficient normalization
%          - 'sine':    Filters are scaled such that a sinusoid with
%                       amplitude A in time domain will exhibit the same
%                       amplitude in the time-frequency representation.
%          - 'impulse': Filters are scaled such that an impulse in time
%                       domain will exhibit a flat response in the
%                       time-frequency representation (in the frame that 
%                       centers the impulse)
%          - 'none':      ...
%     'winfun':        defines the window function that is used for filter
%                   design. See winfuns for more information.
%
%   See also:  nsgtf_real, winfuns
%
%   References:
%     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�rfler. A Matlab 
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
%     Transforms.
%
%     G. A. Velasco, N. Holighaus, M. D�rfler, and T. Grill. Constructing an
%     invertible constant-Q transform with non-stationary Gabor frames.
%     Proceedings of DAFX11, Paris, 2011.
%     
%     N. Holighaus, M. D�rfler, G. Velasco, and T. Grill. A framework for
%     invertible, real-time constant-q transforms. Audio, Speech, and
%     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
%     
%
%
% Copyright (C) 2013 Christian Sch�rkhuber.
% 
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 3.0 Unported 
% License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/3.0/ 
% or send a letter to 
% Creative Commons, 444 Castro Street, Suite 900, 
% Mountain View, California, 94041, USA.

% Authors: Christian Sch�rkhuber
% Date: 20.09.13


%% check input arguments

%defaults
rasterize = 'full'; %fully rasterized
phasemode = 'global';
outputFormat = 'sparse'; %only applies if rasterize == 'octave'
normalize = 'sine';
windowFct = 'hann';
gamma = 0;


if nargin >= 6
    Larg = length(varargin);
    for ii=1:2:Larg
       switch varargin{ii}
           case {'rasterize'}
               rasterize = varargin{ii+1};
           case {'phasemode'}
               phasemode = varargin{ii+1};
           case {'format'}
               outputFormat = varargin{ii+1};
           case {'gamma'}
               gamma = varargin{ii+1};
           case {'normalize'}
               normalize = varargin{ii+1};
           case {'win'}
               windowFct = varargin{ii+1};
       end
    end
end
    

%% window design
[g,shift,M] = nsgcqwin(fmin,fmax,B,fs, length(x), ...
        'winfun', windowFct, 'gamma', gamma, 'fractional', 0);
 fbas = fs*cumsum(shift(2:end))./ length(x);
 fbas = fbas(1:size(M,1)/2-1);
 

%% compute coefficients
bins = size(M,1)/2 - 1;
switch rasterize
    case 'full'
        M(2:bins+1) = M(bins+1);
        M(bins+3:end) = M(bins+1:-1:2);
        
           
    case 'piecewise'
        temp = M(bins+1);
        octs = ceil(log2(fmax/fmin));
        %make sure that the number of coefficients in the highest octave is
        %dividable by 2 at least octs-times
        temp = ceil(temp/2^octs)*2^octs;      
        mtemp = temp./ M;
        mtemp = 2.^( ceil(log2(mtemp)) -1);
        mtemp = temp./ mtemp;
        mtemp(bins+2) = M(bins+2); %don't rasterize Nyquist bin
        mtemp(1) = M(1); %don't rasterize DC bin
        M = mtemp;
       
    otherwise
end

switch normalize
    case {'sine','Sine','SINE','sin'}
        normFacVec = 2*M(1:bins+2)./length(x);
    case {'impulse','Impulse', 'IMPULSE','imp'}
        normFacVec = 2*M(1:bins+2)/cellfun(@length,g);
    case {'none','None','NONE','no'}
        normFacVec = ones(bins+2,1);
    otherwise
        error('Unkown normalization method!');
end

normFacVec = [normFacVec; normFacVec(end-1:-1:2)];
g = arrayfun(@(k) (g{k}*normFacVec(k)),1:(2*bins+2),'UniformOutput',0).'; 

c = nsgtf_real(x,g,shift,M,phasemode);

switch rasterize
    case 'full'
        cDC = cell2mat(c(1)).';   
        cNyq = cell2mat(c(bins+2)).';
        c = cell2mat(c(2:bins+1).');
    case 'piecewise'
        cDC = cell2mat(c(1));   
        cNyq = cell2mat(c(bins+2));
        if strcmp(outputFormat,'sparse')
            c = cqtCell2Sparse(c,M).';
        else
            c = c(2:end-1);
        end
        
    otherwise
        cDC = cell2mat(c(1));   
        cNyq = cell2mat(c(end));
        c = c(2:end-1);
end


%% output
Xcq = struct('c', {c.'}, 'g', {g}, 'shift', shift, 'M', {M}, ...
    'xlen', length(x), 'phasemode', phasemode, 'rast', rasterize, ...
    'fmin', fmin, 'fmax', fmax, 'B', B, 'cDC', cDC, 'cNyq', cNyq, ...
    'format', outputFormat, 'fbas', fbas);


