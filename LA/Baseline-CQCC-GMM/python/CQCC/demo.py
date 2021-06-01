import librosa
from cqcc import cqcc
import matplotlib.pyplot as plt 

# filename = "./D18_1000001.wav"

filename = "./source1.wav"

# INPUT SIGNAL
x,fs = librosa.load(filename, sr = 16000); # from ASVspoof2015 database
x = x.reshape(x.shape[0], 1)  # for one-channel signal 

print(x.shape)
# fs: 16000
# x: (64244,)

# PARAMETERS
B = 96
fmax = fs/2
fmin = fmax/2**9
d = 16
cf = 19
ZsdD = 'ZsdD'

# COMPUTE CQCC FEATURES
CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
print("cqcc_feat:", CQcc.shape)       # number of frames * number of cep
print("cqcc_lpms:", LogP_absCQT.shape)   
#### visulization spectrum #####
plt.figure()
plt.title('Log power magnitude spectrum of CQCC')
# plt.ylabel('Log power magnitude')
# plt.xlabel('Number of frames')
plt.imshow(LogP_absCQT.T)

# COMPUTE MFCC FEATURES
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read(filename)
mfcc_feat = mfcc(sig,rate)      # number of frames * number of cep
fbank_feat = logfbank(sig,rate)
print("mfcc_feat:", mfcc_feat.shape)
print("mfcc_lgms:", fbank_feat.shape)

plt.figure()
plt.title('Log Mel-filterbank energy features of MFCC')
# plt.ylabel('Log power magnitude')
# plt.xlabel('Number of frames')
plt.imshow(fbank_feat.T)
plt.show()
