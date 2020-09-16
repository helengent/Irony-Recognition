function parameters = AMS_init(nFFT_speech,nFFT_ams,nChnl,Srate)

[analys_filter FB2FFT] = gen_analys_filter(nFFT_speech,nChnl,Srate);

% FFT MF->Selected Modulation Frequency Transformation
MF_T = zeros(15,nFFT_ams/2);
MF_T(1,1:3) = [0.4082 0.8165 0.4082];
MF_T(2,2:5) = [0.3162 0.6325 0.6325 0.3162];
MF_T(3,3:6) = [0.3162 0.6325 0.6325 0.3162];
MF_T(4,4:6) = [0.4082 0.8165 0.4082];
MF_T(5,4:7) = [0.3162 0.6325 0.6325 0.3162];
MF_T(6,5:8) = [0.3162 0.6325 0.6325 0.3162];
MF_T(7,6:9) = [0.3162 0.6325 0.6325 0.3162];
MF_T(8,8:10) = [0.4082 0.8165 0.4082];
MF_T(9,9:12) = [0.3162 0.6325 0.6325 0.3162];
MF_T(10,11:13) = [0.4082 0.8165 0.4082];
MF_T(11,13:15) = [0.4082 0.8165 0.4082];
MF_T(12,15:17) = [0.4082 0.8165 0.4082];
MF_T(13,17:19) = [0.4082 0.8165 0.4082];
MF_T(14,20:22) = [0.4082 0.8165 0.4082];
MF_T(15,23:25) = [0.4082 0.8165 0.4082];

win = hanning(nFFT_speech);
win = win(:);

% R = 12; %downsampling rate
% R = 3; is_modified.R = 3;

R = Srate/4000;
is_modified.R = R; % modified by Yang Lu, April 27, 2010

[lp_B, lp_A] = butter(6, 400/Srate*R);
parameters.analys_filter = analys_filter;
parameters.FB2FFT = FB2FFT;
parameters.nFFT_speech = nFFT_speech;
parameters.nFFT_ams = nFFT_ams;
parameters.nChnl = nChnl;
parameters.MF_T = MF_T;
parameters.Srate = Srate;
parameters.step = nFFT_speech/2;
parameters.win = win;
parameters.env_choice = 'abs';
parameters.lp_B = lp_B;
parameters.lp_A = lp_A;
parameters.R = R;

