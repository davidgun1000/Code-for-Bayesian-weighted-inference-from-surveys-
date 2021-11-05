function [lower,upper] = inverted_gamma_draw(alph,beta)
draw = (1./gamrnd(alph,1/beta,10000,1));
lower = quantile(draw,0.025);
upper = quantile(draw,0.975);