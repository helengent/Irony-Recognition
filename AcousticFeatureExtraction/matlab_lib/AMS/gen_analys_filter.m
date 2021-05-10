function [analys_filter FB2FFT] = gen_analys_filter(nFFT,nChnl,Srate)
%%
% Generate the analysis filter bank and 
% the mapping matrix of BM from the channel domain to FFT domain


FB2FFT = zeros(nFFT/2,nChnl);
nOrd = 6;
FS = Srate/2;

[lower cent_freq upper] = mel(nChnl,0,Srate/2);
bandwidth = round(upper - lower);
cent_freq = round(cent_freq);
   
for i = 1:nChnl
    low_f(i) = cent_freq(i) - bandwidth(i)/2;
    up_f(i) = cent_freq(i) + bandwidth(i)/2;
    lower_ind(i) = ceil(low_f(i)/Srate*nFFT);
    upper_ind(i) = ceil(up_f(i)/Srate*nFFT);
    if i>1
        if lower_ind(i)<=upper_ind(i-1)
            lower_ind(i) = upper_ind(i-1) + 1;
        end
    end
    if upper_ind(i)<lower_ind(i)
        upper_ind(i) = lower_ind(i);
    end
    FB2FFT(lower_ind(i):upper_ind(i),i) = 1;%/(upper_ind(i) - lower_ind(i) + 1);
end
% FB2FFT = [FB2FFT;FB2FFT(end:-1:1,:)];
FB2FFT = [FB2FFT;zeros(1,size(FB2FFT,2));FB2FFT(end:-1:2,:)]; % modified by Gibak, Feb.18, 2008
FB2FFT = sparse(FB2FFT);

analys_filter.A = zeros(nChnl,nOrd+1);
analys_filter.B = zeros(nChnl,nOrd+1);
PLOT = 0;
useHigh = 1;
for i = 1:nChnl
    W1 = [low_f(i)/FS, up_f(i)/FS];
    if i == nChnl
        if useHigh == 0
            [b,a] = butter(3,W1);
        else
            [b,a] = butter(6,W1(1),'high');
        end
    else
        [b,a] = butter(3,W1);
    end
    analys_filter.B(i,1:nOrd+1) = b;
    analys_filter.A(i,1:nOrd+1) = a;

    if PLOT == 1
        [h,f] = freqz(b,a,512,Srate);
        plot(f,20*log10(abs(h)+eps));
        set(gca,'Ylim',[-50, 4]);
        %pause
        hold on
    end
end
if PLOT == 1
    hold off
end

