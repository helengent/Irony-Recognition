function ns_ams = extract_AMS_GT(x, fs, nChnl, nb_frames)
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

%%
% Level Adjustment
[x ratio]= LTLAdjust(x, fs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pre-emphasis for speech signal
% cl = filter([1.5 -0.45],1,cl);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
len = floor(4*fs/1000); % 4ms, frame size in samples, envelope length
if rem(len,2)==1
   len = len+1;
end
env_step = 0.25; % 1.00ms or 0.25ms, advance size, envelope step


s_frame_len = 20; %32ms for each frame
nFFT_speech = s_frame_len/1000*fs;
AMS_frame_len = s_frame_len/env_step; % 128 frames of envelope corresponding to 128*0.25 = 32ms
AMS_frame_step = AMS_frame_len/2; % step size

nFFT_env = AMS_frame_len;
nFFT_ams = AMS_frame_len*2;

k = 1;% sample position of the speech signal
kk = 1;
KK = nb_frames;%floor(Nframes/AMS_frame_step) + (AMS_frame_len/AMS_frame_step-1);
ss = 1; % sample position of the noisy speech for synthesize
ns_ams = zeros(nChnl*15,KK);

parameters = AMS_init_FFT(nFFT_env,nFFT_speech,nFFT_ams,nChnl,fs);
parameters_FB = AMS_init(nFFT_speech,20,nChnl,fs); %64 isn't used in this routine

% X_sub = FB_filter(x, parameters_FB); % time domain signals in subbands

%GAMMATONER FILTER OUTPUTS
[~, X_sub, ~, ~] = makeRateMap_IHC(x, fs, 100, 7500, 34, 10,8, 'none','none','none',0);


ENV_x = env_extraction(X_sub, parameters_FB); %time domain envelope in subbands

ns_env = ENV_x;

win_ams = window(@hann,AMS_frame_len);
repwin_ams = repmat(win_ams,1,nChnl);
for kk=1:KK
   start_idx = 1 + (AMS_frame_step*(kk-1));
   end_idx = AMS_frame_len + (AMS_frame_step*(kk-1));
   
   if end_idx<=length(ns_env)
      ns_env_frm = ns_env(:, start_idx:end_idx);
   else
      zero_padding =  zeros(size(ns_env, 1), end_idx - length(ns_env));
      ns_env_frm = [ns_env(:,start_idx:length(ns_env)), zero_padding];
   end
   ams = abs(fft(ns_env_frm'.*repwin_ams,nFFT_ams));
   ams = parameters.MF_T*ams(1:nFFT_ams/2,:);
   ams = ams';
   ns_ams(:,kk) = reshape(ams,[],1);
   
end
