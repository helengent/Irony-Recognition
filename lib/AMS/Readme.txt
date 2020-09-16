
Feature extraction: AMS
Usage:
[ns_ams, true_SNR] = extract_AMS_TrainingData_FFT_FB(noisy_file_name,clean_file_name,number_of_frequency_bands);
noisy_file_name: file name of noisy signal
clean_file_name: file name of clean signal
number_of_frequency_bands: number of mel frequency filter banks
ns_ams: AMS
true_SNR: true SNR calculated using oracle clean signal and noise


Example:
[ns_ams, true_SNR] = extract_AMS_TrainingData_FFT_FB('ns_bab_m5dB_S_01_01.wav','clean_S_01_01.wav',25);

The dimension of 'ns_ams' is (Nbands x 15) x Nframes, where Nbands=25 in the above example.

Note that only the base AMS features (a(t,k)) are computed. The delta AMS features can be easily computed using Eq 2 and 3 [1].

The default sampling rate used in [1] is 12KHz, and the sampling rate could be changed to other values.
The only requirement is that the sampling rate must be no smaller than 4KHz.

Authors: Yang Lu, Gibak Kim

References:
[1] Kim, G., Lu, Y., Hu, Y. and Loizou, P. (2009). "An algorithm that improves speech intelligibility in noise for normal-hearing
 listeners," Journal of Acoustical Society of America, 126(3), 1486-1494