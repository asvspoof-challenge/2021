function gd = nsdual(g,shift,M)
%NSDUAL  Canonical dual NSG frame (for painless systems)
%   Usage: gd = nsdual(g,shift,M)
%
%   Input parameters:
%         g         : Cell array of window functions/filters
%         shift     : Vector of time/frequency shifts
%         M         : Number of frequency channels (vector/scalar)
%   Output parameters:
%         gd        : Dual window functions 
%
%   Given a non-stationary Gabor frame specified by the windows g, shift 
%   parameters shift, and channel numbers M, NSDUAL computes the
%   canonical dual frame windows/filters gd by inverting the diagonal of 
%   the frame operator and applying the inverse to g. More explicitly,
%
%      gd{n} = g{n} / ( sum M(l) |g{l}|^2 ), 
%                        l  
%
%   If g, shift, M specify a painless frame, i.e. 
%   SUPP(G{N})  <= M(n)~forall~n and 
%
%      A <= sum ( M(n) |g{n}|^2 ) <= B, for some 0 < A <= B < infty
%            n  
%
%   the computation will result in the canonical dual frame. If  g, 
%   shift, M specify a frame, but the first condition is violated, the 
%   result can be interpreted as a first approximation of the corresponding 
%   canonical dual frame.
% 
%   Note, the time shifts corresponding to the dual window sequence is the
%   same as the original shift sequence and as such already given.
%
%   If g, shift, M is a painless frame, the output can be used for 
%   perfect reconstruction of a signal using the inverse nonstationary 
%   Gabor transform NSIGT.
% 
%   See also:  nsgt, nsigt, nsgt_real, nsigt_real, nsgtf, nsigtf
% 
%   References:
%     P. Balazs, M. Dörfler, F. Jaillet, N. Holighaus, and G. A. Velasco.
%     Theory, implementation and applications of nonstationary Gabor Frames.
%     J. Comput. Appl. Math., 236(6):1481-1496, 2011.
%     
%
%   Url: http://nsg.sourceforge.net/doc/core_routines/nsdual.php

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

% Author: Nicki Holighaus, Gino Velasco
% Date: 23.04.13

% Check input arguments

if nargin < 3
    for kk = 1:length(shift)
        M(kk) = length(g{kk}); M = M.';
    end
end

if nargin < 2
    error('Not enough input arguments');
end

if max(size(M)) == 1
    M = M(1)*ones(length(shift),1);
end

% Setup the necessary parameters
N = length(shift);

posit = cumsum(shift);
Ls = posit(N);
posit = posit-shift(1);

diagonal=zeros(Ls,1);
win_range = cell(N,1);

% Construct the diagonal of the frame operator matrix explicitly

for ii = 1:N
    Lg = length(g{ii});
    
    win_range{ii} = mod(posit(ii)+(-floor(Lg/2):ceil(Lg/2)-1),Ls)+1;
    diagonal(win_range{ii}) = diagonal(win_range{ii}) + ...
        (fftshift(g{ii}).^2)*M(ii);
end

% Using the frame operator and the original window sequence, compute
% the dual window sequence

gd = g;

for ii=1:N
    gd{ii} = ifftshift(fftshift(gd{ii})./diagonal(win_range{ii}));
end
