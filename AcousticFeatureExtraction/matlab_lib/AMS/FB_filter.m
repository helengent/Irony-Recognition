function sig_sub = FB_filter(sig, parameters)
%%
% Yang Lu
% Jan, 2008
%
%

%%
FB = parameters.analys_filter;
nChnl = parameters.nChnl;

A = FB.A;
B = FB.B;

sig_sub = zeros(nChnl, length(sig));

for n = 1:nChnl
    sig_sub(n,:) = filter(B(n,:),A(n,:),sig);
end