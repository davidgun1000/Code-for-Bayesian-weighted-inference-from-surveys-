function [mu_store,sig2_store]=sample_param_BWE_FPBB(com,n,N)

% CODE TO GENERATE WEIGHTED SAMPLES USING FINITE POPULATION BAYESIAN
% BOOTSTRAP
% THIS CODE REPLICATE THE EXPERIMENT 1 IN GUNAWAN ET AL 2020, BAYESIAN
% WEIGHTED INFERENCE IN SURVEYS
num_FPBB=250;

l_boot=zeros(n,1);
Nnn = (N-n)/n;
df=n-1;

mu_store=zeros(num_FPBB,1);
sig2_store=zeros(num_FPBB,1);

for j = 1:num_FPBB
    
    l_boot=zeros(n,1);
    for k=1:(N-n)
        
         newweights_num=com(:,2)-1+l_boot.*Nnn;
         newweights_den=(N-n)+(k-1)*Nnn;
         newweights=newweights_num./newweights_den;
         [y_select(k,1),idx]=datasample(com(:,1),1,'replace',true,'weights',newweights);
         lk=zeros(n,1);
         lk(idx,1)=1;
         l_boot=l_boot+lk;
    end
     
    FPBB_population=[com(:,1);y_select];
    x=datasample(FPBB_population,n,'replace',true);
     
    y_bar = mean(x);   
    var_hat = var(x);
    sig2_estimate = (1/(gamrnd(df/2,1/((df/2)*var_hat))));
    mu_estimate = normrnd(y_bar,sqrt(sig2_estimate)/sqrt(n));
    mu_store(j,1) = mu_estimate;
    sig2_store(j,1) = sig2_estimate; 
    
end




end