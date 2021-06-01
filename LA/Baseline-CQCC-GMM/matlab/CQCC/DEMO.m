% Usage example of cqcc function
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2016 EURECOM, France.
%
% This work is licensed under the Creative Commons
% Attribution-NonCommercial-ShareAlike 4.0 International
% License. To view a copy of this license, visit
% http://creativecommons.org/licenses/by-nc-sa/4.0/
% or send a letter to
% Creative Commons, 444 Castro Street, Suite 900,
% Mountain View, California, 94041, USA.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% ADD CQT TOOLBOX TO THE PATH
addpath('CQT_toolbox_2013');

%% INPUT SIGNAL
[x,fs] = audioread('D18_1000001.wav'); % from ASVspoof2015 database

%% PARAMETERS
B = 96;
fmax = fs/2;
fmin = fmax/2^9;
d = 16;
cf = 19;
ZsdD = 'ZsdD';

%% COMPUTE CQCC FEATURES
[CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
    cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD);
