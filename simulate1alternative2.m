%simulation 2
%function out = simulate1alternative2(k_P)

%this code replicates the experiment 1 in Gunawan et al (2020)
%The code compares the weighted frequentist and Bayesian approaches. 
%Please read the paper for further details. 

k_P='1';
seednum = str2double(k_P);
parpool(28)
% %seed = randi([1,500]);
rng(seednum);
%seed =123;
bigloop = 250;
mean_mu_store_UMLE = zeros(bigloop,1);
var_mu_store_UMLE = zeros(bigloop,1);
mean_sig2_store_UMLE = zeros(bigloop,1);
var_sig2_store_UMLE = zeros(bigloop,1);

mean_mu_store_PMLE = zeros(bigloop,1);
var_mu_store_PMLE = zeros(bigloop,1);
mean_sig2_store_PMLE = zeros(bigloop,1);
var_sig2_store_PMLE = zeros(bigloop,1);

mean_mu_store_UBE = zeros(bigloop,1);
var_mu_store_UBE = zeros(bigloop,1);
mean_sig2_store_UBE = zeros(bigloop,1);
var_sig2_store_UBE = zeros(bigloop,1);

mean_mu_store_BPPE = zeros(bigloop,1);
var_mu_store_BPPE = zeros(bigloop,1);
mean_sig2_store_BPPE = zeros(bigloop,1);
var_sig2_store_BPPE = zeros(bigloop,1);

mean_mu_store_BWE_multinomial = zeros(bigloop,1);
var_mu_store_BWE_multinomial = zeros(bigloop,1);
mean_sig2_store_BWE_multinomial = zeros(bigloop,1);
var_sig2_store_BWE_multinomial = zeros(bigloop,1);

mean_mu_store_BWE_FPBB = zeros(bigloop,1);
var_mu_store_BWE_FPBB = zeros(bigloop,1);
mean_sig2_store_BWE_FPBB = zeros(bigloop,1);
var_sig2_store_BWE_FPBB = zeros(bigloop,1);

mu_inside_UBE = 0;
sig2_inside_UBE = 0;

mu_inside_UMLE = 0;
sig2_inside_UMLE = 0;

mu_inside_PMLE = 0;
sig2_inside_PMLE = 0;

mu_inside_BPPE = 0;
sig2_inside_BPPE = 0;

mu_inside_BWE_multinomial = 0;
sig2_inside_BWE_multinomial = 0;

mu_inside_BWE_FPBB = 0;
sig2_inside_BWE_FPBB = 0;



parfor s = 1:bigloop
tic
    N = 100000;
true_mu_y = 10;
true_sig2_y = 100;
true_corr = 0.8;
true_mu_x = 0;
true_sig2_x = 9;
true_cov = true_corr*sqrt(true_sig2_y)*sqrt(true_sig2_x);
r = mvnrnd([true_mu_y true_mu_x],[true_sig2_y true_cov; true_cov true_sig2_x],N);
x1 = r(:,2);
beta0 = -1.8;
beta1 = 0.1;
X = [ones(N,1),r(:,2)];
beta = [beta0 ; beta1];
X_beta = X*beta;
selection = normrnd(X_beta,1);
ind = (selection >=0);
ind_0 = (ind==0);
x_0 = x1(ind_0,1);
y_0 = r(ind_0,1);
ind_1 = (ind==1);
x_1 = x1(ind_1,1);
y_1 = r(ind_1,1);

n = sum(ind);
dep_selected_sample = r(ind,1);
x1_selected_sample = x1(ind,1);
prob_selected_sample = normcdf(beta0 + beta1*x1_selected_sample,0,1);
weight_selected_sample= 1./prob_selected_sample;
standardise_weight_selected_sample = weight_selected_sample.*(n/sum(weight_selected_sample));
%----------------------------------------------------------------------------------------------------------------
%standardise the weight
%do the inference

normalise_standardise_weight_selected_sample = standardise_weight_selected_sample./sum(standardise_weight_selected_sample);
% com = [dep_selected_sample standardise_weight_selected_sample normalise_standardise_weight_selected_sample];
% com_sort = sortrows(com,1);
% com_sort(:,4) = cumsum(com_sort(:,3));
% com_sort(n,4) = 1;

com_multinomial = [dep_selected_sample standardise_weight_selected_sample normalise_standardise_weight_selected_sample];
com=com_multinomial;
com_FPBB = [dep_selected_sample,weight_selected_sample];
com_sort = sortrows(com_multinomial,1);
com_sort(:,4) = cumsum(com_sort(:,3));
com_sort(n,4) = 1;


%Bayesian Approach
df = n-1;
sample_mean = sum(com_sort(:,1))/n;
sample_variance = sum(((com_sort(:,1)-sample_mean).^2))./df;
weighted_sample_mean = sum(com_sort(:,1).*com_sort(:,2))/sum(com_sort(:,2));

weighted_sample_variance = sum(com_sort(:,2).*((com_sort(:,1)-weighted_sample_mean).^2))./df;
%nit = 1000;
%----------------------------------------------------------------------------
%unweighted case
  
     mu_store_UBE = sample_mean;
     var_mu_UBE = (sample_variance/n)*(df/(df-2));
     lower_bound_mu_UBE = mu_store_UBE - sqrt(sample_variance/n)*tinv(0.975,df);
     upper_bound_mu_UBE = mu_store_UBE + sqrt(sample_variance/n)*tinv(0.975,df);
     alph_UBE = df/2;
     beta_UBE = (df/2)*sample_variance;
     sig2_store_UBE = beta_UBE/(alph_UBE-1);
     var_sig2_UBE = (beta_UBE^2)/((alph_UBE-1)^2*(alph_UBE-2));
     [lower_bound_sig2_UBE,upper_bound_sig2_UBE] = inverted_gamma_draw(alph_UBE,beta_UBE);
     if (true_mu_y>=lower_bound_mu_UBE) & (true_mu_y<=upper_bound_mu_UBE)
        mu_inside_UBE = mu_inside_UBE+1;
     end

     if (true_sig2_y>=lower_bound_sig2_UBE) & (true_sig2_y<=upper_bound_sig2_UBE)
        sig2_inside_UBE = sig2_inside_UBE+1;
     end

     mean_mu_store_UBE(s,1) = mu_store_UBE;
     var_mu_store_UBE(s,1) = var_mu_UBE;
     mean_sig2_store_UBE(s,1) = sig2_store_UBE;
     var_sig2_store_UBE(s,1) = var_sig2_UBE;
     

%-----------------------------------------------------------------------------

%----------------------------------------------------------------------------
%The Bayesian Pseudo Likelihood
   
    mu_store_BPPE = weighted_sample_mean;    
    var_mu_BPPE = (weighted_sample_variance/n)*(df/(df-2));
    lower_bound_mu_BPPE = mu_store_BPPE - sqrt(weighted_sample_variance/n)*tinv(0.975,df);
    upper_bound_mu_BPPE = mu_store_BPPE + sqrt(weighted_sample_variance/n)*tinv(0.975,df);
    alph_BPPE = df/2;
    beta_BPPE = (df/2)*weighted_sample_variance;
    sig2_store_BPPE = beta_BPPE/(alph_BPPE-1);
    var_sig2_BPPE = (beta_BPPE^2)/((alph_BPPE-1)^2*(alph_BPPE-2));
    [lower_bound_sig2_BPPE,upper_bound_sig2_BPPE] = inverted_gamma_draw(alph_BPPE,beta_BPPE);
    if (true_mu_y>=lower_bound_mu_BPPE) & (true_mu_y<=upper_bound_mu_BPPE)
        mu_inside_BPPE = mu_inside_BPPE +1;
    end

    if (true_sig2_y>=lower_bound_sig2_BPPE) & (true_sig2_y<=upper_bound_sig2_BPPE)
        sig2_inside_BPPE = sig2_inside_BPPE+1;
    end

    mean_mu_store_BPPE(s,1) = mu_store_BPPE;
    var_mu_store_BPPE(s,1) = var_mu_BPPE;
    mean_sig2_store_BPPE(s,1) = sig2_store_BPPE;
    var_sig2_store_BPPE(s,1) = var_sig2_BPPE;
    
%-----------------------------------------------------------------------------
%the weighted algorithm BWE multinomial

  %tic
%   for i = 1:nit
%      x = datasample(com(:,1),n,'Replace',true,'Weights',com(:,3));
%      y_bar_BWE = mean(x);   
%      var_hat_BWE = var(x);
%      sig2_estimate_BWE = (1/(gamrnd(df/2,1/((df/2)*var_hat_BWE))));
%      mu_estimate_BWE = normrnd(y_bar_BWE,sqrt(sig2_estimate_BWE)/sqrt(n));
%      mu_store_BWE(i,1) = mu_estimate_BWE;
%      sig2_store_BWE(i,1) = sig2_estimate_BWE;   
%   end
%   %toc

[mu_store_BWE_multinomial,sig2_store_BWE_multinomial]=sample_param_BWE_multinomial(com_multinomial,n);
lower_bound_mu_BWE_multinomial = quantile(mu_store_BWE_multinomial,0.025);
upper_bound_mu_BWE_multinomial = quantile(mu_store_BWE_multinomial,0.975);
lower_bound_sig2_BWE_multinomial =quantile(sig2_store_BWE_multinomial,0.025);
upper_bound_sig2_BWE_multinomial =quantile(sig2_store_BWE_multinomial,0.975);
if (true_mu_y>=lower_bound_mu_BWE_multinomial) & (true_mu_y<=upper_bound_mu_BWE_multinomial)
    mu_inside_BWE_multinomial = mu_inside_BWE_multinomial +1;
end
 
if (true_sig2_y>=lower_bound_sig2_BWE_multinomial) & (true_sig2_y<=upper_bound_sig2_BWE_multinomial)
    sig2_inside_BWE_multinomial = sig2_inside_BWE_multinomial+1;
end
 
mean_mu_store_BWE_multinomial(s,1) = mean(mu_store_BWE_multinomial);
var_mu_store_BWE_multinomial(s,1) = var(mu_store_BWE_multinomial);
mean_sig2_store_BWE_multinomial(s,1) = mean(sig2_store_BWE_multinomial);
var_sig2_store_BWE_multinomial(s,1) = var(sig2_store_BWE_multinomial);
%-----------------------------------------------------------------------------

%-----------------------------------------------------------------------------
%the weighted algorithm BWE multinomial

  %tic
%   for i = 1:nit
%      x = datasample(com(:,1),n,'Replace',true,'Weights',com(:,3));
%      y_bar_BWE = mean(x);   
%      var_hat_BWE = var(x);
%      sig2_estimate_BWE = (1/(gamrnd(df/2,1/((df/2)*var_hat_BWE))));
%      mu_estimate_BWE = normrnd(y_bar_BWE,sqrt(sig2_estimate_BWE)/sqrt(n));
%      mu_store_BWE(i,1) = mu_estimate_BWE;
%      sig2_store_BWE(i,1) = sig2_estimate_BWE;   
%   end
%   %toc

[mu_store_BWE_FPBB,sig2_store_BWE_FPBB]=sample_param_BWE_FPBB(com_FPBB,n,N);
lower_bound_mu_BWE_FPBB = quantile(mu_store_BWE_FPBB,0.025);
upper_bound_mu_BWE_FPBB = quantile(mu_store_BWE_FPBB,0.975);
lower_bound_sig2_BWE_FPBB =quantile(sig2_store_BWE_FPBB,0.025);
upper_bound_sig2_BWE_FPBB =quantile(sig2_store_BWE_FPBB,0.975);
if (true_mu_y>=lower_bound_mu_BWE_FPBB) & (true_mu_y<=upper_bound_mu_BWE_FPBB)
    mu_inside_BWE_FPBB = mu_inside_BWE_FPBB +1;
end
 
if (true_sig2_y>=lower_bound_sig2_BWE_FPBB) & (true_sig2_y<=upper_bound_sig2_BWE_FPBB)
    sig2_inside_BWE_FPBB = sig2_inside_BWE_FPBB+1;
end
 
mean_mu_store_BWE_FPBB(s,1) = mean(mu_store_BWE_FPBB);
var_mu_store_BWE_FPBB(s,1) = var(mu_store_BWE_FPBB);
mean_sig2_store_BWE_FPBB(s,1) = mean(sig2_store_BWE_FPBB);
var_sig2_store_BWE_FPBB(s,1) = var(sig2_store_BWE_FPBB);
%-----------------------------------------------------------------------------


%The Frequentist approach
%PMLE (Pseudo Maximum Likelihood Estimation)
mu_PMLE = sum(com(:,2).*com(:,1))./sum(com(:,2));

sig2_PMLE = sum(com(:,2).*((com(:,1)-mu_PMLE).^2))./sum(com(:,2));

part1_21_PMLE = -(com(:,2).*com(:,1))./(sig2_PMLE^2);
part2_21_PMLE = com(:,2).*(mu_PMLE/(sig2_PMLE^2));
hes_21_PMLE = sum(part1_21_PMLE+part2_21_PMLE);
hes_11_PMLE = sum(-com(:,2)./sig2_PMLE);
part1_22_PMLE = com(:,2)./(2*(sig2_PMLE^2));
part2_22_PMLE = -(com(:,2)./(sig2_PMLE^3)).*((com(:,1)-mu_PMLE).^2);
hes_22_PMLE = sum(part1_22_PMLE + part2_22_PMLE);
hes_dagun_PMLE = [hes_11_PMLE 0; 0 hes_22_PMLE];
inv_hes_dagun_PMLE = inv(hes_dagun_PMLE); 
% %grad_mu_sq = sum(((com(:,2).*com(:,1))./sig2 - (mu/sig2).*com(:,2)).^2);
grad_mu_sq_PMLE = sum(((com(:,2)./sig2_PMLE).*(com(:,1)-mu_PMLE)).^2);
grad_sig2_sq_PMLE = sum(((com(:,2)./(2*(sig2_PMLE^2))).*((com(:,1)-mu_PMLE).^2 - sig2_PMLE)).^2);
cross_prod_grad_mu_sig_PMLE = sum(((com(:,2).*com(:,1))./sig2_PMLE - (mu_PMLE/sig2_PMLE).*com(:,2)).*((com(:,2)./(2*(sig2_PMLE^2))).*((com(:,1)-mu_PMLE).^2 - sig2_PMLE)));
grad_hat_PMLE = [grad_mu_sq_PMLE cross_prod_grad_mu_sig_PMLE; cross_prod_grad_mu_sig_PMLE grad_sig2_sq_PMLE];
cov_weighted_PMLE = inv_hes_dagun_PMLE*(grad_hat_PMLE)*inv_hes_dagun_PMLE;
mean_mu_store_PMLE(s,1) = mu_PMLE;
% %mean_mu_hat_store(s,1) = mu_hat;
var_mu_store_PMLE(s,1) = cov_weighted_PMLE(1,1);
mean_sig2_store_PMLE(s,1) = sig2_PMLE;
var_sig2_store_PMLE(s,1) = cov_weighted_PMLE(2,2);
stderr1_PMLE = sqrt(cov_weighted_PMLE(1,1));
stderr2_PMLE = sqrt(cov_weighted_PMLE(2,2));

lower_bound_mu_PMLE = mu_PMLE-1.96*stderr1_PMLE;
upper_bound_mu_PMLE = mu_PMLE+1.96*stderr1_PMLE;
% 
if (true_mu_y>=lower_bound_mu_PMLE) & (true_mu_y<=upper_bound_mu_PMLE)
    mu_inside_PMLE = mu_inside_PMLE +1;
end
% 
lower_bound_sig2_PMLE = sig2_PMLE-1.96*stderr2_PMLE;
upper_bound_sig2_PMLE = sig2_PMLE+1.96*stderr2_PMLE;
% 
if (true_sig2_y>=lower_bound_sig2_PMLE) & (true_sig2_y<=upper_bound_sig2_PMLE)
     sig2_inside_PMLE = sig2_inside_PMLE+1;
end

%-----------------------------------------------------------------------------------------------------------------------------------------
%Unweighted MLE
mu_UMLE = mean(dep_selected_sample);
sig2_UMLE = var(dep_selected_sample);
hes_11_UMLE = (-n./sig2_UMLE);
part1_22_UMLE = 1./(2*(sig2_UMLE^2));
part2_22_UMLE = -(1./(sig2_UMLE^3)).*(((dep_selected_sample-mu_UMLE).^2));
hes_22_UMLE = sum(part1_22_UMLE + part2_22_UMLE);
hes_dagun_UMLE = [hes_11_UMLE 0; 0 hes_22_UMLE];
I_UMLE = -hes_dagun_UMLE;
cov_unweighted_UMLE = inv(I_UMLE);
mean_mu_store_UMLE(s,1) = mu_UMLE;
var_mu_store_UMLE(s,1) = cov_unweighted_UMLE(1,1);
mean_sig2_store_UMLE(s,1) = sig2_UMLE;
var_sig2_store_UMLE(s,1) = cov_unweighted_UMLE(2,2);
stderr1_UMLE = sqrt(cov_unweighted_UMLE(1,1));
stderr2_UMLE = sqrt(cov_unweighted_UMLE(2,2));
lower_bound_mu_UMLE = mu_UMLE-1.96*stderr1_UMLE;
upper_bound_mu_UMLE = mu_UMLE+1.96*stderr1_UMLE;
lower_bound_sig2_UMLE = sig2_UMLE-1.96*stderr2_UMLE;
upper_bound_sig2_UMLE = sig2_UMLE+1.96*stderr2_UMLE;
if (true_mu_y>=lower_bound_mu_UMLE) & (true_mu_y<=upper_bound_mu_UMLE)
    mu_inside_UMLE = mu_inside_UMLE +1;
end

if (true_sig2_y>=lower_bound_sig2_UMLE) & (true_sig2_y<=upper_bound_sig2_UMLE)
    sig2_inside_UMLE = sig2_inside_UMLE+1;
end
toc
end
%save('output_unweighted_Bayes_rho0.8sig100.mat','mean_mu_store','var_mu_store','mean_sig2_store','var_sig2_store','mu_inside','sig2_inside')

save('sim1_large_08_100.mat','mean_mu_store_UBE','var_mu_store_UBE','mu_inside_UBE','mean_sig2_store_UBE','var_sig2_store_UBE','sig2_inside_UBE',...
     'mean_mu_store_BPPE','var_mu_store_BPPE','mu_inside_BPPE','mean_sig2_store_BPPE','var_sig2_store_BPPE','sig2_inside_BPPE',...
     'mean_mu_store_BWE_multinomial','var_mu_store_BWE_multinomial','mu_inside_BWE_multinomial','mean_sig2_store_BWE_multinomial','var_sig2_store_BWE_multinomial','sig2_inside_BWE_multinomial',...
     'mean_mu_store_PMLE','var_mu_store_PMLE','mu_inside_PMLE','mean_sig2_store_PMLE','var_sig2_store_PMLE','sig2_inside_PMLE',...
     'mean_mu_store_UMLE','var_mu_store_UMLE','mu_inside_UMLE','mean_sig2_store_UMLE','var_sig2_store_UMLE','sig2_inside_UMLE',...
     'mean_mu_store_BWE_FPBB','var_mu_store_BWE_FPBB','mu_inside_BWE_FPBB','mean_sig2_store_BWE_FPBB','var_sig2_store_BWE_FPBB','sig2_inside_BWE_FPBB');


% nm_UBE = ['output_large_sample_paper_UBE_0.2_100',k_P,'.mat']
% save(nm_UBE,'mean_mu_store_UBE','var_mu_store_UBE','mu_inside_UBE','mean_sig2_store_UBE','var_sig2_store_UBE','sig2_inside_UBE')
% 
% nm_BPPE = ['output_large_sample_paper_BPPE_0.2_100',k_P,'.mat']
% save(nm_BPPE,'mean_mu_store_BPPE','var_mu_store_BPPE','mu_inside_BPPE','mean_sig2_store_BPPE','var_sig2_store_BPPE','sig2_inside_BPPE')   
%     
% nm_BWE = ['output_large_sample_paper_BWE_0.2_100',k_P,'.mat']
% save(nm_BWE,'mean_mu_store_BWE','var_mu_store_BWE','mu_inside_BWE','mean_sig2_store_BWE','var_sig2_store_BWE','sig2_inside_BWE')
% 
% nm_PMLE = ['output_large_sample_paper_PMLE_0.2_100',k_P,'.mat']
% save(nm_PMLE,'mean_mu_store_PMLE','var_mu_store_PMLE','mu_inside_PMLE','mean_sig2_store_PMLE','var_sig2_store_PMLE','sig2_inside_PMLE')
% 
% nm_UMLE = ['output_large_sample_paper_UMLE_0.2_100',k_P,'.mat']
% save(nm_UMLE,'mean_mu_store_UMLE','var_mu_store_UMLE','mu_inside_UMLE','mean_sig2_store_UMLE','var_sig2_store_UMLE','sig2_inside_UMLE')
toc








%------------------------------------------------------------------------------------------








%estimating probit model by using ML

% [estimator] = probit_ML(ind,x1,x2,x3);
% 
% beta0_hat = estimator(1,1);
% beta1_hat = estimator(1,2);


%weighted bootstrap

% for i = 1:nit
%      for j = 1:n
%        u1 = rand();
%        index = find(u1 <= com_sort(:,5),1,'first');
%        new_data(j,1) = com_sort(index(1,1),1);    
%        new_weight(j,1) = com_sort(index(1,1),2);
%      end
%      x = new_data;
%      new_weight = new_weight;
%      y_bar = sum(x.*new_weight)./sum(new_weight);
%      y_bar_store(i,1) = y_bar;
%     
%     
% end

%     lower_bound_sig2 = quantile(sig2_store,0.025);
%     upper_bound_sig2 = quantile(sig2_store,0.975);
%    var_mu_store = (df/(df-2))*
%    sig2_estimate = (1/(gamrnd(df/2,1/((df/2)*var_hat))));
%    mu_estimate = normrnd(y_bar,sqrt(sig2_estimate)/sqrt(n));
%    ybar_t = y_bar + sqrt(var_hat/n)*trnd(df); 
%    mu_store(i,1) = mu_estimate;
%    sig2_store(i,1) = sig2_estimate;   
%    y_bar_t(i,1) = ybar_t;
%    y_bar_store(i,1) = y_bar;
% end











