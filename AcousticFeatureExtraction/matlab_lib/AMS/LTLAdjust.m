function [x ratio]= LTLAdjust(x, Srate)



max_val = max(x);
x = x/max_val*0.4;
ratio = 0.4/max_val;