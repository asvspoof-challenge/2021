function [x gd] = icqt(Xcq)
%ICQT  inverse Constant-Q/Variable-Q transform
%   Usage:  [x gd] = icqt(Xcq, varargin)
%
%   Input parameters:
%         Xcq       : struct obtained by cqt(...)
%         varargin  : Optional input pairs (see table below)
%
%   Output parameters: 
%         x         : reconstructed time domain signal
%         gd        : synthesis filterbank
%
%   See also:  cqt, nsigtf_real, winfuns
%
%   References:
%     C. Schörkhuber, A. Klapuri, N. Holighaus, and M. Dörfler. A Matlab 
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
%     Transforms.
%
%     G. A. Velasco, N. Holighaus, M. Dörfler, and T. Grill. Constructing an
%     invertible constant-Q transform with non-stationary Gabor frames.
%     Proceedings of DAFX11, Paris, 2011.
%     
%     N. Holighaus, M. Dörfler, G. Velasco, and T. Grill. A framework for
%     invertible, real-time constant-q transforms. Audio, Speech, and
%     Language Processing, IEEE Transactions on, 21(4):775-785, April 2013.
%     
%
%
% Copyright (C) 2013 Christian Schörkhuber.
% 
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 3.0 Unported 
% License. To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/3.0/ 
% or send a letter to 
% Creative Commons, 444 Castro Street, Suite 900, 
% Mountain View, California, 94041, USA.

% Authors: Christian Schörkhuber
% Date: 20.09.13
%% window design
Xcq.gd = {nsdual(Xcq.g,Xcq.shift,Xcq.M)}; % compute dual filters.
% Xcq.gd = {Xcq.g}; %temp;

switch Xcq.rast
    case 'piecewise'
        if strcmp(Xcq.format, 'sparse')
            c = cqtSparse2Cell(Xcq.c,Xcq.M, Xcq.cDC, Xcq.cNyq); 
        else
            c = cell(1,size(Xcq.c,2)+2);
            c(2:end-1) = Xcq.c(1:end);
            c(1) = {Xcq.cDC};
            c(end) = {Xcq.cNyq};
        end
    case 'full'
        c = cell(1,size(Xcq.c,1)+2);
        c(2:end-1) = num2cell(Xcq.c.',1);
        c(1) = {Xcq.cDC.'};
        c(end) = {Xcq.cNyq.'};
        
    otherwise
        c = cell(1,size(Xcq.c,2)+2);
        c(2:end-1) = Xcq.c(1:end);
        c(1) = {Xcq.cDC};
        c(end) = {Xcq.cNyq};
end


x = nsigtf_real(c,Xcq.gd{1},Xcq.shift,Xcq.xlen, Xcq.phasemode);

gd = Xcq.gd{1};