function [weightmat] = TryOptimize(weightmat,exp_num_trans,regCoeff,features)

    options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','Derivativecheck','on','FinDiffType','central');
    fun = @(weightmat)fandg(weightmat,exp_num_trans,regCoeff,features);
    [weightmat,fval,exitflag] = fminunc(fun, weightmat, options);
    exitflag
    fval
    %weightmat = w;

end

function [f1,g1] = fandg(weightmat,exp_num_trans, regCoeff,features)

s = size(weightmat);
Q = s(1);
D = s(2);

transmat = exp(weightmat * features')';
%transmat = exp_wdotf(weightmat ,features')';
%transmat = mk_stochastic(transmat);
transmat = bsxfun(@rdivide, transmat, sum(transmat,2));

if(s(1)~=Q)
    error('Something wrong here. globalfeatures should be D x nex and weights should be QxD');
end

f1 = 0;
g1 = zeros(Q,D);

temp = log(transmat) .* exp_num_trans;
f1 = sum(sum(temp));
%temp = 0.1 * ones(Q,D);
f1 = (-1)* f1 + (regCoeff * (norm(weightmat,1)));

% Compute gradient
d = diag(sum(exp_num_trans,2));
g1 = (exp_num_trans' - (transmat'* d)) * features;
[maxv, ind] = max(sum(weightmat.*sign(weightmat)));
temp = zeros(Q,D);
%signmat = (sign(weightmat(:,ind)) + ones(Q,1))/2;
temp(:,ind) = sign(weightmat(:,ind));
g1 = (-1)* g1 ;
g1 = g1 + regCoeff * temp;
end


