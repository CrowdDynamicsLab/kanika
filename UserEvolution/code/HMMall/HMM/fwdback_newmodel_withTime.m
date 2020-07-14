function [alpha, beta, gamma, loglik, xi_summed] = fwdback_newmodel_withTime(init_state_distrib, ...
   features, weightmat, obslik, varargin)
% FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
%
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
%
% Notation:
% Y(t) = observation, Q(t) = hidden state, M(t) = mixture variable (for MOG outputs)
% A(t) = discrete input (action) (for POMDP models)
%
% INPUT:
% init_state_distrib(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
%  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
% obslik(i,t) = Pr(Y(t)| Q(t)=i)
%   (Compute obslik using eval_pdf_xxx on your data sequence first.)
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% For HMMs with MOG outputs: if you want to compute gamma2, you must specify
% 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
% 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []
%  or mixmat{t}(m,q) if not stationary
%
% For HMMs with discrete inputs:
% 'act' - act(t) = action performed at step t
%
% Optional arguments:
% 'fwd_only' - if 1, only do a forwards pass and set beta=[], gamma2=[]  [0]
% 'scaled' - if 1,  normalize alphas and betas to prevent underflow [1]
% 'maximize' - if 1, use max-product instead of sum-product [0]
%
% OUTPUTS:
% alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
% beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
% gamma(i,t) = p(Q(t)=i | y(1:T))
% loglik = log p(y(1:T))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
% xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
% gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
%
% If fwd_only = 1, these become
% alpha(i,t) = p(Q(t)=i | y(1:t))
% beta = []
% gamma(i,t) = p(Q(t)=i | y(1:t))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:t))
% gamma2 = []
%
% Note: we only compute xi if it is requested as a return argument, since it can be very large.
% Similarly, we only compute gamma2 on request (and if using MOG outputs).
%
% Examples:
%
% [alpha, beta, gamma, loglik] = fwdback(pi, A, multinomial_prob(sequence, B));
%
% [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat);
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(pi, A, B, 'obslik2', B2, 'mixmat', mixmat);

if 0 % nargout >= 5
  warning('this now returns sum_t xi(i,j,t) not xi(i,j,t)')
end

if nargout >= 5, compute_xi = 1; else compute_xi = 0; end
if nargout >= 6, compute_gamma2 = 1; else compute_gamma2 = 0; end

[obslik2, fwd_only, scaled, act, maximize, compute_xi, compute_gamma2] = ...
   process_options(varargin, ...
       'obslik2', [], ...
       'fwd_only', 0, 'scaled', 1, 'act', [], 'maximize', 0, ...
                   'compute_xi', compute_xi, 'compute_gamma2', compute_gamma2);

[Q T] = size(obslik);
D = size(weightmat,2);

%%if isempty(act)
%%act = ones(1,T);
%%transmat = { transmat } ;
%%end

%% Create Fmatrice (Not needed now)
% Featuremat = {};
% 
% for t = 1:T-1
%     Featuremat{t} = zeros(Q,N);
%      for q = 1:Q
%          for q_ = 1:Q
%              if( q ~= q_)
%                 Featuremat{t}(q,q_) = 0;
%              else
%                  Featuremat{t}(q,q_) = 1;
%              end
%          end
%      end
%      Featuremat{t}(q,Q+1) = t;
% end
     
                 


%%
scale = ones(1,T);

% scale(t) = Pr(O(t) | O(1:t-1)) = 1/c(t) as defined by Rabiner (1989).
% Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T))
% or log P = sum_t log scale(t).
% Rabiner suggests multiplying beta(t) by scale(t), but we can instead
% normalise beta(t) - the constants will cancel when we compute gamma.

loglik = 0;

alpha = zeros(Q,T);
gamma = zeros(Q,T);
if compute_xi
 xi_summed = zeros(Q,Q,T);
else
 xi_summed = [];
end

%%%%%%%%% Forwards %%%%%%%%%%

t = 1;
alpha(:,1) = init_state_distrib(:) .* obslik(:,t);
if scaled
 %[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
 [alpha(:,t), scale(t)] = normalise(alpha(:,t));
end
%assert(approxeq(sum(alpha(:,t)),1))
for t=2:T   % for all Tju
 %trans = transmat(:,:,act(t-1))';
 %%trans = transmat{act(t-1)};
 transmat = zeros(Q,Q);
 for i = 1:Q
     features(i,D) = t/T;
 end
 for i = 1:Q
        transmat(i,:) = (exp(weightmat * features(i,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
 end

 transmat = mk_stochastic(transmat);
 
 m = transmat' * alpha(:,t-1);
 alpha(:,t) = m(:) .* obslik(:,t);
 
 if scaled
   %[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
   [alpha(:,t), scale(t)] = normalise(alpha(:,t));
 end
 
 if compute_xi && fwd_only  % useful for online EM
   %xi(:,:,t-1) = normaliseC((alpha(:,t-1) * obslik(:,t)') .* trans);
   xi_summed = xi_summed + normalise((alpha(:,t-1) * obslik(:,t)') .* transmat);
 end
 %assert(approxeq(sum(alpha(:,t)),1))
end

if scaled
 if any(scale==0)
   loglik = -inf;
 else
   loglik = sum(log(scale));
 end
else
 loglik = log(sum(alpha(:,T)));
end

if fwd_only
 gamma = alpha;
 beta = [];
 return;
end

%%%%%%%%% Backwards %%%%%%%%%%

beta = zeros(Q,T);

beta(:,T) = ones(Q,1);
%gamma(:,T) = normaliseC(alpha(:,T) .* beta(:,T));
gamma(:,T) = normalise(alpha(:,T) .* beta(:,T));

for t=T-1:-1:1
    
 transmat = zeros(Q,Q);
 for i = 1:Q
     features(i,D) = t/T;
 end
 for i = 1:Q
        transmat(i,:) = (exp(weightmat * features(i,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
 end

 transmat = mk_stochastic(transmat);
 
 b = beta(:,t+1) .* obslik(:,t+1);
 %trans = transmat(:,:,act(t));
 %trans = transmat{act(t)};
 trans = transmat;
 
 beta(:,t) = trans * b;

 if scaled
   %beta(:,t) = normaliseC(beta(:,t));
   beta(:,t) = normalise(beta(:,t));
 end
 %gamma(:,t) = normaliseC(alpha(:,t) .* beta(:,t));
 gamma(:,t) = normalise(alpha(:,t) .* beta(:,t));
 if compute_xi
   %xi(:,:,t) = normaliseC((trans .* (alpha(:,t) * b')));
   xi_summed(:,:,t) = normalise((alpha(:,t) * b'));
 end
 
end

%%
% We now explain the equation for gamma2
% Let zt=y(1:t-1,t+1:T) be all observations except y(t)
% gamma2(Q,M,t) = P(Qt,Mt|yt,zt) = P(yt|Qt,Mt,zt) P(Qt,Mt|zt) / P(yt|zt)
%                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|zt) / P(yt|zt)
% Now gamma(Q,t) = P(Qt|yt,zt) = P(yt|Qt) P(Qt|zt) / P(yt|zt)
% hence
% P(Qt,Mt|yt,zt) = P(yt|Qt,Mt) P(Mt|Qt) [P(Qt|yt,zt) P(yt|zt) / P(yt|Qt)] / P(yt|zt)
%                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|yt,zt) / P(yt|Qt)
