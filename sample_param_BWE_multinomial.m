function [mu_store,sig2_store]=sample_param_BWE_multinomial(com,n)

% CODE TO GENERATE WEIGHTED SAMPLES USING BWE-EDF
% THIS CODE REPLICATE THE EXPERIMENT 1 IN GUNAWAN ET AL 2020, BAYESIAN
% WEIGHTED INFERENCE IN SURVEYS
df = n-1;
num_FPBB=250;

mu_store=zeros(num_FPBB,1);
sig2_store=zeros(num_FPBB,1);

for j = 1:num_FPBB
     x = datasample(com(:,1),n,'Replace',true,'Weights',com(:,3));
     y_bar = mean(x);   
     var_hat = var(x);
     sig2_estimate = (1/(gamrnd(df/2,1/((df/2)*var_hat))));
     mu_estimate = normrnd(y_bar,sqrt(sig2_estimate)/sqrt(n));
     mu_store(j,1) = mu_estimate;
     sig2_store(j,1) = sig2_estimate;   
end

end

%    for i=1:s
 %    end
     %mu_store=[mu_store;mu_temp];
     %sig2_store=[sig2_store;sig2_temp];
 