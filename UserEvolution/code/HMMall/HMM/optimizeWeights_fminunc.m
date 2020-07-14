function [weights] = optimizeWeights_fminunc(weights, globalFeatures,exp_count_q_c0_sumt_ex,regCoeff);
    %   options = optimoptions('fminunc');
    options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','Display','iter','DerivativeCheck','on');
    fun = @(weights)fandg(weights, globalFeatures,exp_count_q_c0_sumt_ex,regCoeff);
    [w,fval] = fminunc(fun, weights, options);
    weights = w;

end

function [f1,g1] = fandg(weights, globalFeatures,exp_count_q_c0_sumt_ex, regCoeff)
%weight is QxD
%globalFeat is Dx nex
%exp_count_q_c0_sumt_ex is Q x nex
% disp('computing func and grad');
s = size(weights);
Q = s(1);
D = s(2);
s = size(globalFeatures);
nex = s(2);
if(s(1)~=D)
    error('Something wrong here. globalfeatures should be D x nex and weights should be QxD');
end

f1 = 0;
g1 = zeros(Q,D);

% Compute function
prob = exp(weights * globalFeatures);
prob = bsxfun(@rdivide, prob, sum(prob));
f1 = sum(sum(log(prob) .* exp_count_q_c0_sumt_ex));
f1 = (-1)* f1 + regCoeff*(norm(weights,2)^2);

% Compute gradient
d = diag(sum(exp_count_q_c0_sumt_ex));
g1=(exp_count_q_c0_sumt_ex - (prob * d))* globalFeatures';
g1 = (-1)* g1 + regCoeff * 2 * weights;

end

function [ret] = exp_wdotf(w,f)
% return exp(w.f)
if(length(w)~=length(f))
    disp('lengths of w and f should be same!!!');
end
ret = exp(dot(w,f));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% f = 0;
% g = zeros(Q,D);
% for l = 1:nex
%     lambda = exp_count_q_c0_sumt_ex(:,l); % lambda is Qx1
%     feat = globalFeatures(:,l)'; %feat is 1xD
%     
%     % Compute the function
%     prob = exp(sum(weights .* repmat(feat,[Q,1]),2)); % prob is Q x 1
%     prob = mk_stochastic(prob);
%     logprob = log(prob);
%     f = f + sum(logprob .* lambda);
%     
%     % Compute the gradient
%     sumLambdas = sum(lambda); 
%     g = g + (lambda(:) - (prob(:) * sumLambdas)) * feat;
% end
% f = (-1)* f + regCoeff*(norm(weights,2)^2);
% g = (-1)* g + regCoeff * 2 * weights;

%%%%%%%%%% Following is INCORRECT. Want to minimize -f
function [f,g] = fandg_old(weights, globalFeatures,exp_count_q_c0_sumt_ex, regCoeff)
%weight is QxD
%globalFeat is Dx nex
%exp_count_q_c0_sumt_ex is Q x nex
% disp('computing func and grad');
s = size(weights);
Q = s(1);
D = s(2);
s = size(globalFeatures);
nex = s(2);
if(s(1)~=D)
    error('Something wrong here. globalfeatures should be D x nex and weights should be QxD');
end

f = 0;
for l = 1:nex
    for i = 1:Q
        prob(i) = exp_wdotf(weights(i,:),globalFeatures(:,l));
    end
    prob = mk_stochastic(prob);
    logprob = log(prob);
    f = f + sum(logprob(:) .* exp_count_q_c0_sumt_ex(:,l));
end
f = f + regCoeff*(norm(weights,2)^2);

g = zeros(Q,D);
for l = 1:nex
    sumLambdas = sum(exp_count_q_c0_sumt_ex(:,l));
    for i = 1:Q
        lambda(i) = exp_count_q_c0_sumt_ex(i,l);
        temp = globalFeatures(:,l) .* (lambda(i) - prob(i) * sumLambdas);
        g(i,:) = g(i,:) + temp';
    end
end     
g = g + regCoeff .* 2 .* weights;
end

