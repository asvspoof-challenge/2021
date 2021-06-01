function [stat,delta,double_delta]=lfcc_bp(speech,Fs,Window_Length,NFFT,No_Filter,No_coeff,low_freq,high_freq) 
% Function for computing LFCC features 
% Usage: [stat,delta,double_delta]=extract_lfcc(file_path,Fs,Window_Length,No_Filter) 
%
% Input: file_path=Path of the speech file
%        Fs=Sampling frequency in Hz
%        Window_Length=Window length in ms
%        NFFT=No of FFT bins
%        No_Filter=No of filter
%        low_freq=minimum frequency for LFCC processing
%        high_freq=maximum frequency for LFCC processing
%
% Output: stat=Static LFCC (Size: NxNo_Filter where N is the number of frames)
%        delta=Delta LFCC (Size: NxNo_Filter where N is the number of frames)
%        double_delta=Double Delta LFCC (Size: NxNo_Filter where N is the number of frames)
%
%        Written by Md Sahidullah at School of Computing, University of
%        Eastern Finland (email: sahid@cs.uef.fi)
%        
%        Implementation details are available in the following paper:
%        M. Sahidullah, T. Kinnunen, C. Hanilçi, A comparison of features 
%        for synthetic speech detection, Proc. Interspeech 2015, 
%        pp. 2087--2091, Dresden, Germany, September 2015.

%---------------------------FRAMING & WINDOWING----------------------------
frame_length_inSample=(Fs/1000)*Window_Length;
framedspeech=buffer(speech,frame_length_inSample,frame_length_inSample/2,'nodelay')';
w=hamming(frame_length_inSample);
y_framed=framedspeech.*repmat(w',size(framedspeech,1),1);
%--------------------------------------------------------------------------
f=(Fs/2)*linspace(0,1,NFFT/2+1);
filbandwidthsf=linspace(low_freq,high_freq,No_Filter+2);
fr_all=(abs(fft(y_framed',NFFT))).^2;

[~,closestIndex_low_freq] = min(abs(f-low_freq));
[~,closestIndex_high_freq] = min(abs(f-high_freq));
fa_all=fr_all(closestIndex_low_freq:closestIndex_high_freq,:)';
filterbank=zeros(size(fa_all,2),No_Filter);
f = f(closestIndex_low_freq:closestIndex_high_freq);

for i=1:No_Filter
    filterbank(:,i)=trimf(f,[filbandwidthsf(i),filbandwidthsf(i+1),...
        filbandwidthsf(i+2)]);
end
filbanksum=fa_all*filterbank(1:end,:);
%-------------------------Calculate Static Cepstral------------------------
t=dct(log10(filbanksum'+eps));
t=(t(1:No_coeff,:));
stat=t'; 
delta=Deltas(stat',1)';
double_delta=Deltas(delta',1)';
%--------------------------------------------------------------------------

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
D = D(:,hlen*2+1:end);
D = D./(2*sum((1:hlen).^2));
end