function fr = nsigtf_real(c,g,shift,Ls,phasemode)
% NSIGTF_REAL  Nonstationary Gabor filterbank synthesis for real signals
%   Usage: fr = nsigtf_real(c,g,shift,M,Ls)
%
%   Input parameters: 
%         c         : Cell array of nonstationary Gabor coefficients
%         g         : Cell array of synthesis filters
%         shift     : Vector of time shifts
%         M         : Number of time channels (vector/scalar)
%         Ls        : Length of the analyzed signal
%         phasemode  :  can be set to (default is 'global')
%                      - 'local':     Zero-centered filtered used
%                      - 'global':    Mapping function used (see reference)
%   Output parameters:
%         fr        : Synthesized real-valued signal (Channels are stored 
%                     in the columns)
%
%   Given the cell array c of nonstationary Gabor filterbank 
%   coefficients, a set of filters g and frequency shifts shift, this 
%   function computes the corresponding nonstationary Gabor filterbank
%   synthesis for real valued signals. 
%
%   Note that, due to the structure of the coefficient array in the real
%   valued setting, all entries g{n} with N > length(c) will be ignored
%   and assumed to be fully supported on the negative frequencies.
%
%   Let P(n)=sum_{l=1}^{n} shift(l), then the synthesis formula reads:
%
%                    N-1 
%       fr_temp(l) = sum sum c{n}(m)g{n}[l-P(n)]*exp(-2*pi*i*(l-P(n))*m/M(n)),
%                    n=0  m
%   
%   for l=0,cdots,Ls-1.  In practice, the synthesis formula is realized 
%   by fft and overlap-add. To synthesize the negative frequencies, 
%   fr_temp is truncated to length floor( Ls/2 )+1. Afterwards 
%   ifftreal implicitly computes the hermite symmetric extension and 
%   computes the inverse Fourier transform, i.e. fr = ifftreal(fr_temp).
% 
%   If a nonstationary Gabor frame was used to produce the coefficients 
%   and g is a corresponding dual frame, this function should perfectly 
%   reconstruct the originally analyzed signal to numerical precision.
%   
%   Multichannel output will save each channel in a column of fr.
%
%   See also:  nsdual, nstight
% 
%   References:
%     C. Schörkhuber, A. Klapuri, N. Holighaus, and M. Dörfler. A Matlab 
%     Toolbox for Efficient Perfect Reconstruction log-f Time-Frequecy 
%     Transforms.
%
%     P. Balazs, M. Dörfler, F. Jaillet, N. Holighaus, and G. A. Velasco.
%     Theory, implementation and applications of nonstationary Gabor Frames.
%     J. Comput. Appl. Math., 236(6):1481-1496, 2011.
%     
%     G. A. Velasco, N. Holighaus, M. DÃ¶rfler, and T. Grill. Constructing an
%     invertible constant-Q transform with non-stationary Gabor frames.
%     Proceedings of DAFX11, Paris, 2011.
%     

% Copyright (C) 2013 Nicki Holighaus.
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
% Edited by Christian Schörkhuber, 25.09.2013

%Check input arguments
if nargin < 4
    error('Not enough input arguments');
end

if iscell(c) == 0 % If matrix format coefficients were used, convert to
    % cell
    if ndims(c) == 2
        [N,chan_len] = size(c); CH = 1;
        c = mat2cell(c.',chan_len,ones(1,N)).';
    else
        [N,chan_len,CH] = size(c);
        ctemp = mat2cell(permute(c,[2,1,3]),chan_len,ones(1,N),ones(1,CH));
        c = permute(ctemp,[2,3,1]);
        clear ctemp;
    end
else
    [CH, N] = size(c);
end

posit = cumsum(shift);      % Calculate positions from shift vector
NN = posit(end);            % Reconstruction length before truncation
posit = posit-shift(1);   % Adjust positions

fr = zeros(NN,CH); % Initialize output

% The overlap-add procedure including multiplication with the synthesis
% windows

for ii = 1:N
    Lg = length(g{ii});
  
    win_range = mod(posit(ii)+(-floor(Lg/2):ceil(Lg/2)-1),NN)+1;
    
    temp = fft(c{ii},[],1)*length(c{ii});
    
    if strcmp(phasemode,'global')
        %shift the center frequency back to baseband prior to
        %reconstruction (after having them shifted to the 'true alias
        %frequency')
        fsNewBins = size(c{ii},1);
        fkBins = posit(ii);
        displace = fkBins - floor(fkBins/fsNewBins) * fsNewBins;
        temp = circshift(temp, -displace);
    end
    
    temp = temp(mod([end-floor(Lg/2)+1:end,1:ceil(Lg/2)]-1,...
        length(temp))+1,:);
    
    fr(win_range,:) = fr(win_range,:) + ...
        bsxfun(@times,temp,g{ii}([Lg-floor(Lg/2)+1:Lg,1:ceil(Lg/2)]));
end

nyqBin = floor(Ls/2) + 1;
fr(nyqBin+1:end) = conj( fr(nyqBin  - (~logical(mod(Ls,2))) : -1 : 2) );
fr = real(ifft(fr)); %this should be real already - but just to remove any 
                     %imaginary part that occurs due to limited numerical
                     %presicion.
