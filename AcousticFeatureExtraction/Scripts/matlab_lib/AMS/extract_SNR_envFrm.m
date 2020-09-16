function SNR = extract_SNR_envFrm(cl_env,tn_env,len)
% cl = sqrt(cl_env);
cl = cl_env;
cl = sum(cl');
% tn = sqrt(tn_env);
tn = tn_env;
tn = sum(tn');
% SNR = 10*log10((cl.^2)./(tn.^2)+eps);
SNR = 10*log10((cl)./(tn)+eps);


