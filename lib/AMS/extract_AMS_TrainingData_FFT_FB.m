function [ns_ams, true_SNR] = extract_AMS_TrainingData_FFT_FB(filename, cl_file, nChnl)
% filename: file name of waveform for extracting feature
% cl_file: file name of clean signal
% nChnl: # of channels (filterbank)

% Order of AMS = 15
% ns_ams: AMS (N x # of frames, N=Order of AMS x # of channel)
% true_SNR: true SNR (# of channel x # of frames)
%

%
% Yang Lu & Gibak Kim
% April, 2007
%
% This program is used to extract the AMS from a file as well as the SNRs 
% in subbands with high efficiency (frame by frame).


%% read waveform
% noisy speech
[x Srate] = wavread(filename);
% true noise
[cl Srate] = wavread(cl_file);
% clean speech
tn = x - cl;
%% 
% Level Adjustment
[x ratio]= LTLAdjust(x, Srate);
tn = tn*ratio;
cl = cl*ratio;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-emphasis for speech signal
% cl = filter([1.5 -0.45],1,cl);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 
len = floor(4*Srate/1000); % 4ms, frame size in samples, envelope length
if rem(len,2)==1
    len = len+1; 
end
env_step = 0.25; % 1.00ms or 0.25ms, advance size, envelope step
len2 = floor(env_step*Srate/1000); 
Nframes = floor(length(x)/len2)-len/len2+1;
Srate_env = 1/(env_step/1000); % Since we calculate the envelope every 0.25ms, the sampling rate for envelope is this.
% win = hanning(len);
win = window(@hann,len);
s_frame_len = 32; %32ms for each frame


nFFT_speech = s_frame_len/1000*Srate;
AMS_frame_len = s_frame_len/env_step; % 128 frames of envelope corresponding to 128*0.25 = 32ms
AMS_frame_step = AMS_frame_len/2; % step size

nFFT_env = AMS_frame_len;
nFFT_ams = AMS_frame_len*2;

k = 1;% sample position of the speech signal
kk = 1;
KK = floor(Nframes/AMS_frame_step) - (AMS_frame_len/AMS_frame_step-1);
ss = 1; % sample position of the noisy speech for synthesize
ns_ams = zeros(nChnl*15,KK);
true_SNR = zeros(nChnl,KK);

parameters = AMS_init_FFT(nFFT_env,nFFT_speech,nFFT_ams,nChnl,Srate);
parameters_FB = AMS_init(nFFT_speech,64,nChnl,Srate); %64 isn't used in this routine

X_sub = FB_filter(x, parameters_FB); % time domain signals in subbands
TN_sub = FB_filter(tn, parameters_FB);
CL_sub = FB_filter(cl, parameters_FB);

ENV_x = env_extraction(X_sub, parameters_FB); %time domain envelope in subbands
ENV_tn = env_extraction(TN_sub, parameters_FB);
ENV_cl = env_extraction(CL_sub, parameters_FB);

ns_env = ENV_x;
tn_env = ENV_tn;
cl_env = ENV_cl;

win_ams = window(@hann,AMS_frame_len);
repwin_ams = repmat(win_ams,1,nChnl);
for kk=1:KK
    start_idx = 1 + (AMS_frame_step*(kk-1));
    end_idx = AMS_frame_len + (AMS_frame_step*(kk-1));
    
    if end_idx<=length(ns_env)
    	ns_env_frm = ns_env(:, start_idx:end_idx);
        tn_env_frm = tn_env(:, start_idx:end_idx);
        cl_env_frm = cl_env(:, start_idx:end_idx);
    else
        zero_padding =  zeros(size(ns_env, 1), end_idx - length(ns_env));
        ns_env_frm = [ns_env(:,start_idx:length(ns_env)), zero_padding];
        tn_env_frm = [tn_env(:,start_idx:length(ns_env)), zero_padding];
        cl_env_frm = [cl_env(:,start_idx:length(ns_env)), zero_padding];
    end
	ams = abs(fft(ns_env_frm'.*repwin_ams,nFFT_ams));
	ams = parameters.MF_T*ams(1:nFFT_ams/2,:);
	ams = ams';
	ns_ams(:,kk) = reshape(ams,[],1);

	
	
	true_SNR(:,kk) = extract_SNR_envFrm(cl_env_frm.^2,tn_env_frm.^2,AMS_frame_len);% calculate SNR from envelope
end

%%
return;
