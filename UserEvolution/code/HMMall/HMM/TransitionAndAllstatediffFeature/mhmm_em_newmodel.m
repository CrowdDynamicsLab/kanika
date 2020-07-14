function [LL, prior, transmat, weightmat, mu, Sigma] = ...
     mhmm_em_newmodel(data, prior, transmat, weightmat, features, mu, Sigma, regCoeff,varargin)
% LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.
% [ll_trace, prior, weightmat, mu, sigma, mixmat] = learn_mhmm(data, ...
%   prior0, weightmat0, mu0, sigma0, mixmat0, ...) 
%
% Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
%
% INPUTS:
% data{ex}(:,t) or data(:,t,ex) if all sequences have the same length
% prior(i) = Pr(Q(1) = i), 
% transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
% mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k ]
% Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
% mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to [] or ones(Q,1) if only one mixture component
%
% Optional parameters may be passed as 'param_name', param_value pairs.
% Parameter names are shown below; default values in [] - if none, argument is mandatory.
%
% 'max_iter' - max number of EM iterations [10]
% 'thresh' - convergence threshold [1e-4]
% 'verbose' - if 1, print out loglik at every iteration [1]
% 'cov_type' - 'full', 'diag' or 'spherical' ['full']
%
% To clamp some of the parameters, so learning does not change them:
% 'adj_prior' - if 0, do not change prior [1]
% 'adj_trans' - if 0, do not change transmat [1]
% 'adj_mix' - if 0, do not change mixmat [1]
% 'adj_mu' - if 0, do not change mu [1]
% 'adj_Sigma' - if 0, do not change Sigma [1]
%
% If the number of mixture components differs depending on Q, just set  the trailing
% entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
% then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.

if ~isempty(varargin) & ~isstr(varargin{1}) % catch old syntax
  error('optional arguments should be passed as string/value pairs')
end

[max_iter, thresh, verbose, cov_type,  adj_prior, adj_trans, adj_mu, adj_Sigma] = ...
    process_options(varargin, 'max_iter', 30, 'thresh', 1e-4, 'verbose', 1, ...
		    'cov_type', 'full', 'adj_prior', 1, 'adj_trans', 1, ...
		    'adj_mu', 1, 'adj_Sigma', 1);
  
previous_loglik = -inf;
loglik = 0;
converged = 0;
num_iter = 1;
LL = [];

if ~iscell(data)
  data = num2cell(data, [1 2]); % each elt of the 3rd dim gets its own cell
end
numex = length(data);


O = size(data{1},1);
Q = length(prior);


while (num_iter <= max_iter) & ~converged
  % E step
  [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op] = ...
      ess_mhmm(prior, features, weightmat, mu, Sigma, data);
  
  
  % M step
  if adj_prior
    prior = normalise(exp_num_visits1);
  end
 
  if adj_mu | adj_Sigma
    [mu2, Sigma2] = mixgauss_Mstep(postmix, m, op, ip, 'cov_type', cov_type);
    if adj_mu
      mu = reshape(mu2, [O Q]);
    end
    if adj_Sigma
      Sigma = reshape(Sigma2, [O O Q]);
    end
  end
  
  if adj_trans
    % transition matrix should be updated on the basis of new wi's and F
    %transmat = mk_stochastic(exp_num_trans);
    
    % OPTIMIZE WEIGHTS
    exp_num_trans = mk_stochastic(exp_num_trans);
    num = (Q*(Q-1))/2;
    diffmu = zeros(1,num);
    k = 1;
    for i = 1:Q-1
        for j = i+1:Q
            diffmu(1,k) = norm((mu(:,i)- mu(:,j)),2)^2;
            k = k+1;
        end
    end

    for i = 1:Q
        features(i,i) = 1;
        features(i,Q+1:Q+num) = diffmu;
    end
    %exp_num_trans = QXQXT
     [weightmat] = optimizeWeights(weightmat,exp_num_trans,regCoeff,features);
     %[weightmat] = optimizeWeights_fminunc(weightmat, features', exp_num_trans,regCoeff);
    for i = 1:Q
        transmat(i,:) = (exp(weightmat * features(i,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
    end

    transmat = mk_stochastic(transmat);
  end
  
  if verbose, fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik); end
  weightmat;
  exp_num_trans;
  num_iter =  num_iter + 1;
  converged = em_converged(loglik, previous_loglik, thresh);
  previous_loglik = loglik;
  LL = [LL loglik];
end


%%%%%%%%%

function [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op] = ...
    ess_mhmm(prior, features, weightmat, mu, Sigma, data)
% ESS_MHMM Compute the Expected Sufficient Statistics for a MOG Hidden Markov Model.
%
% Outputs:
% exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j| Obs(l))
% exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))
%
% Let w(i,k,t,l) = P(Q(t)=i, M(t)=k | Obs(l))
% where Obs(l) = Obs(:,:,l) = O_1 .. O_T for sequence l
% Then 
% postmix(i,k) = sum_l sum_t w(i,k,t,l) (posterior mixing weights/ responsibilities)
% m(:,i,k)   = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)
% ip(i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)' * Obs(:,t,l)
% op(:,:,i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l) * Obs(:,t,l)'


verbose = 0;

%[O T numex] = size(data);
numex = length(data);
O = size(data{1},1);
Q = length(prior);
exp_num_trans = zeros(Q,Q);
exp_num_visits1 = zeros(Q,1);
postmix = zeros(Q,1);
m = zeros(O,Q);
op = zeros(O,O,Q);
ip = zeros(Q,1);


loglik = 0;
if verbose, fprintf(1, 'forwards-backwards example # '); end
for ex=1:numex % each sequence
  if verbose, fprintf(1, '%d ', ex); end
  %obs = data(:,:,ex);
  obs = data{ex};
  T = size(obs,2);
  
  B = mixgauss_prob(obs, mu, Sigma);
  [alpha, beta, gamma,  current_loglik, xi_summed] = fwdback_newmodel(prior, features, weightmat, B);
      
  loglik = loglik +  current_loglik; 
  if verbose, fprintf(1, 'll at ex %d = %f\n', ex, loglik); end
  
  exp_num_trans =  exp_num_trans + xi_summed; % sum(xi,3);
  exp_num_visits1 = exp_num_visits1 + gamma(:,1);
  
  postmix = postmix + sum(gamma,2); 
  gamma2 = reshape(gamma, [Q T]); % gamma2(i,m,t) = gamma(i,t)
  
  for i=1:Q
      w = reshape(gamma2(i,:), [1 T]); % w(t) = w(i,k,t,l)
      wobs = obs .* repmat(w, [O 1]); % wobs(:,t) = w(t) * obs(:,t)
      m(:,i) = m(:,i) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
      op(:,:,i) = op(:,:,i) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
      ip(i) = ip(i) + sum(sum(wobs .* obs, 2)); % ip = sum_t w(t) * obs(:,t)' * obs(:,t)
  end

end
if verbose, fprintf(1, '\n'); end
