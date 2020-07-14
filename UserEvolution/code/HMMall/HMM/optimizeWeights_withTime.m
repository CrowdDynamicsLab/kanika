function [weightmat] = optimizeWeights_withTime(weightmat, exp_num_trans,regCoeff,features)
    %   options = optimoptions('fminunc');
    % 'Algorithm','trust-region',
    % ,'TolX',1e-24,'DiffMinChange',1e-3,'MaxIter',30 'GradObj','on','Derivativescheck','on',,'Display','iter'
    options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','DiffMinChange',1e-3);
    fun = @(weightmat)fandg(weightmat,exp_num_trans,regCoeff,features);
    [w,fval,exitflag] = fminunc(fun, weightmat, options);
    %exitflag
    fval
    weightmat = w;

end

function [f1,g1] = fandg(weightmat,exp_num_trans, regCoeff,features)
%weight is QxD
%globalFeat is Dx nex
%exp_count_q_c0_sumt_ex is Q x nex
% disp('computing func and grad');
s = size(weightmat);
Q = s(1);
D = s(2);
T = size(exp_num_trans,3);
%transmat = exp(weightmat * features')';
%transmat = mk_stochastic(transmat);
%transmat = bsxfun(@rdivide, transmat, sum(transmat));

if(s(1)~=Q)
    error('Something wrong here. globalfeatures should be D x nex and weights should be QxD');
end
regCoeff = 0;
f1 = 0;
g1 = zeros(Q,D);
f1_temp = zeros(Q,Q);
g1_temp = zeros(Q,D);
%T = 25;
for t = 1:T-1
    transmat = zeros(Q,Q);
    for i = 1:Q
        features(i,D) = t/T;
    end
    
    transmat = exp(weightmat * features')';
    transmat = mk_stochastic(transmat);
    exp_num_trans(:,:,t) = mk_stochastic(exp_num_trans(:,:,t));
    
    temp = log(transmat) .* exp_num_trans(:,:,t);
    f1_temp = f1_temp + temp;
    
    exp_num_trans_curr = exp_num_trans(:,:,t);
    d = diag(sum(exp_num_trans_curr,2));
    temp2 = (exp_num_trans_curr' - (transmat'* d)) * features;
    g1_temp = g1_temp + temp2 ;
    
end
f1 = sum(sum(f1_temp));
f1 = (-1)* f1 + regCoeff * (norm(weightmat,1));

% Compute gradient
[maxv, ind] = max(sum(weightmat.*sign(weightmat),2));
temp3 = zeros(Q,D);
temp3(ind,:) = sign(weightmat(ind,:));
g1 = (-1) * g1_temp + regCoeff * temp3;

end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% prob = exp(weights * globalFeatures);
% prob = bsxfun(@rdivide, prob, sum(prob));
% f1 = sum(sum(log(prob) .* exp_count_q_c0_sumt_ex));
% d = diag(sum(exp_count_q_c0_sumt_ex));
% g1=(exp_count_q_c0_sumt_ex - (prob * d))* globalFeatures';


% Compute function
function [ret] = exp_wdotf(w,f)
% return exp(w.f)
if(length(w)~=length(f))
    disp('lengths of w and f should be same!!!');
end
ret = exp(dot(w,f));
end

