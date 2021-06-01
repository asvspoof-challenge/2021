function  [CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
    cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)

%   Constant Q cepstral coefficients
%   Usage:  CQcc = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
%
%   Input parameters:
%         x        : input signal
%         fs       : sampling frequency
%         B        : number of bins per octave [default = 96]
%         fmax     : highest frequency to be analyzed [default = Nyquist frequency]
%         fmin     : lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
%         d        : number of uniform samples in the first octave [default 16]
%         cf       : number of cepstral coefficients excluding 0'th coefficient [default 19]
%         ZsdD     : any sensible combination of the following  [default ZsdD]:
%                      'Z'  include 0'th order cepstral coefficient
%                      's'  include static coefficients (c)
%                      'd'  include delta coefficients (dc/dt)
%                      'D'  include delta-delta coefficients (d^2c/dt^2)
%
%   Output parameters:
%         CQcc              : constant Q cepstral coefficients (nCoeff x nFea)
%         LogP_absCQT       : log power magnitude spectrum of constant Q trasform
%         TimeVec           : time at the centre of each frame [sec]
%         FreqVec           : center frequencies of analysis filters [Hz]
%         Ures_LogP_absCQT  : uniform resampling of LogP_absCQT
%         Ures_FreqVec      : uniform resampling of FreqVec [Hz]
%
%   See also:  cqt
%
%
%   References:
%     M. Todisco, H. Delgado, and N. Evans. A New Feature for Automatic
%     Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients.
%     Proceedings of ODYSSEY - The Speaker and Language Recognition
%     Workshop, 2016.
%
%     C. Sch�rkhuber, A. Klapuri, N. Holighaus, and M. D�fler. A Matlab
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy
%     Transforms. Proceedings AES 53rd Conference on Semantic Audio, London,
%     UK, Jan. 2014. http://www.cs.tut.fi/sgn/arg/CQT/
%
%     G. A. Velasco, N. Holighaus, M. D�fler, and T. Grill. Constructing an
%     invertible constant-Q transform with non-stationary Gabor frames.
%     Proceedings of DAFX11, Paris, 2011.
%
%     N. Holighaus, M. D�fler, G. Velasco, and T. Grill. A framework for
%     invertible, real-time constant-q transforms. Audio, Speech, and
%     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Authors: Massimiliano Todisco {todisco [at] eurecom [dot] fr}
%          Hector Delgado {delgado [at] eurecom [dot] fr}
%
% Version: 1.0
% Date: 22.01.16
%
% User are requested to cite the following paper in papers which report 
% results obtained with this software package.	
%
%     M. Todisco, H. Delgado, and N. Evans. A New Feature for Automatic
%     Speaker Verification Anti-Spoofing: Constant Q Cepstral Coefficients.
%     Proceedings of ODYSSEY - The Speaker and Language Recognition
%     Workshop, 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% CHECK INPUT PARAMETERS
if nargin < 2
    warning('Not enough input arguments.'), return
end

%%% DEFAULT INPUT PARAMETERS
if nargin < 3; B = 96; end
if nargin < 4; fmax = fs/2; end
if nargin < 5; oct = ceil(log2(fmax/20)); fmin = fmax/2^oct; end
if nargin < 6; d = 16; end
if nargin < 7; cf = 19; end
if nargin < 8; ZsdD = 'ZsdD'; end
gamma = 228.7*(2^(1/B)-2^(-1/B));

%%% CQT COMPUTING
Xcq = cqt(x, B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma);

%%% LOG POWER SPECTRUM
absCQT = abs(Xcq.c);
TimeVec = (1:size(absCQT,2))*Xcq.xlen/size(absCQT,2)/fs;
FreqVec = fmin*(2.^((0:size(absCQT,1)-1)/B));
LogP_absCQT = log(absCQT.^2 + eps);

%%% UNIFORM RESAMPLING
kl = (B*log2(1+1/d));
[Ures_LogP_absCQT, Ures_FreqVec] = resample(LogP_absCQT,...
    FreqVec,1/(fmin*(2^(kl/B)-1)),1,1,'spline');

%%% DCT
CQcepstrum = dct(Ures_LogP_absCQT);

%%% DYNAMIC COEFFICIENTS
if strfind(ZsdD, 'Z'); scoeff = 1; else scoeff = 2; end
CQcepstrum_temp = CQcepstrum(scoeff:cf+1,:);
f_d = 3; % delta window size
if strcmp(strrep(ZsdD,'Z',''), 'sdD')
    CQcc = [CQcepstrum_temp; Deltas(CQcepstrum_temp,f_d); ...
        Deltas(Deltas(CQcepstrum_temp,f_d),f_d)];
elseif strcmp(strrep(ZsdD,'Z',''), 'sd')
    CQcc = [CQcepstrum_temp; Deltas(CQcepstrum_temp,f_d)];
elseif strcmp(strrep(ZsdD,'Z',''), 'sD')
    CQcc = [CQcepstrum_temp; Deltas(Deltas(CQcepstrum_temp,f_d),f_d)];
elseif strcmp(strrep(ZsdD,'Z',''), 's')
    CQcc = CQcepstrum_temp;
elseif strcmp(strrep(ZsdD,'Z',''), 'd')
    CQcc = Deltas(CQcepstrum_temp,f_d);
elseif strcmp(strrep(ZsdD,'Z',''), 'D')
    CQcc = Deltas(Deltas(CQcepstrum_temp,f_d),f_d);
elseif strcmp(strrep(ZsdD,'Z',''), 'dD')
    CQcc = [Deltas(CQcepstrum_temp,f_d); Deltas(Deltas(CQcepstrum_temp,f_d),f_d)];
end
end

function D = Deltas(x,hlen)

% Delta and acceleration coefficients
%
% Reference:
%   Young S.J., Evermann G., Gales M.J.F., Kershaw D., Liu X., Moore G., Odell J., Ollason D.,
%   Povey D., Valtchev V. and Woodland P., The HTK Book (for HTK Version 3.4) December 2006.

win = hlen:-1:-hlen;
xx = [repmat(x(:,1),1,hlen),x,repmat(x(:,end),1,hlen)];
D = filter(win, 1, xx, [], 2);
% D = D(:,hlen+1:(end - hlen));
D = D(:,hlen*2+1:end);
D = D./(2*sum((1:hlen).^2));
end
