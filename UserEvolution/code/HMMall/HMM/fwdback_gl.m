function [exp_count_qi_qj_sumt, exp_count_c_sumt, exp_count_q_c1_t1, ...
    exp_count_q_c0_sumt,exp_count_q_t, loglik]= fwdback_gl(pi, transmat, obslik,...
    globalProb, mixPref)
% FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
%
% [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
%
% Notation:
% O(t) = observation, Q(t) = hidden state
%
% INPUT:
% init_state_distrib(i) = Pr(Q(1) = i)
% transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
%  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
% obslik(i,t) = Pr(Y(t)| Q(t)=i)
%   (Compute obslik using eval_pdf_xxx on your data sequence first.)
%
% OUTPUTS:
% alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
% beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
% gamma(i,t) = p(Q(t)=i | y(1:T))
% loglik = log p(y(1:T))
% xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
% xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
% gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)

[Q T] = size(obslik);
C=2;

% disp('observation likelihood is (QxT):')
% obslik %=[0.1,0.2,0.1;0.2,0.5,0.1]
% disp('globalProb is(Qx1):')
% globalProb

%%%%%%%%% Forwards %%%%%%%%%%
alphaTOP = zeros(Q,T); % for c=0
alphaBOT = zeros(Q,T); % for c=1
scalingFactor = zeros(T,1);

t = 1;
alphaTOP(1:Q,t) = globalProb(:) .* obslik(:,t) .* mixPref(1);% for c = 0
alphaBOT(1:Q,t)= pi(:) .* obslik(:,t) .* mixPref(2); % for c = 1
% scaling
scalingFactor(t) = sum(alphaTOP(:,t)) + sum(alphaBOT(:,t));
alphaTOP(1:Q,t) = alphaTOP(1:Q,t) ./ scalingFactor(t);
alphaBOT(1:Q,t) = alphaBOT(1:Q,t) ./ scalingFactor(t);

for t=2:T
    temp = alphaTOP(:,t-1) + alphaBOT(:,t-1);
    alphaTOP(:,t) = obslik(:,t) .* globalProb(:) .* mixPref(1) .* sum(temp);
    alphaBOT(:,t) = obslik(:,t) .* mixPref(2) .* sum(transmat(:,:) .* repmat(temp,[1,Q]))';
    % scaling
    scalingFactor(t) = sum(alphaTOP(:,t)) + sum(alphaBOT(:,t));
    alphaTOP(:,t) = alphaTOP(:,t) ./ scalingFactor(t);
    alphaBOT(:,t) = alphaBOT(:,t) ./ scalingFactor(t);
end
clearvars temp;
%alpha = [alphaTOP;alphaBOT];
%p_obs = sum(alphaTOP(:,T)) + sum(alphaBOT(:,T)); % P(obs sequence ie O_{1..T})
%loglik = log(sum(alpha(:,T)));

if any(scalingFactor==0)
    loglik = -inf;
else
    loglik = sum(log(scalingFactor));
end
 
%%%%%%%%% Backwards %%%%%%%%%%

beta = zeros(Q,T);
beta(:,T) = ones(Q,1);

temp1 = (repmat(globalProb',[Q,1]).*mixPref(1)) + (transmat .* mixPref(2));
for t=T-1:-1:1
    temp2 = beta(:,t+1) .* obslik(:,t+1);
    beta(:,t) = temp1 * temp2;
    beta(:,t) = beta(:,t) ./ scalingFactor(t+1); % scaling
end
clearvars temp1 temp2;

% Sanity check for no scaling: beta at t=0 should correspond to p_obs
% Note that t=0 has only 1 row and not Q rows because it coresponds to an
% imaginary start state which can take only one value and can move to other
% Q states with prior probabilities.
%beta_0 = sum( beta(:,1) .* obslik(:,1) .* (globalProb(:).* mixPref(1)+pi(:).*mixPref(2)));
% if((p_obs-beta_0)>1e-20)
%     p_obs
%     beta_0
%     error('something wrong here. sum of last column of alpha and first column of beta donot match');
% end

% exp_count_qi_qj_sumt(i,j) = sum_{t=1}^{T-1} P(q_t=i, q_{t+1}=j, c_{t+1}=1 | obs)
exp_count_qi_qj_sumt = zeros(Q,Q);
temp1 = mixPref(2) .* (alphaTOP + alphaBOT);
temp2 = beta .* obslik;
for t=1:T-1
    temp=transmat .* repmat(temp1(:,t),[1,Q]) .* repmat(temp2(:,t+1)',[Q,1]);
    temp = temp ./scalingFactor(t+1);
    exp_count_qi_qj_sumt= exp_count_qi_qj_sumt + temp;
end
%exp_count_qi_qj_sumt = exp_count_qi_qj_sumt ./ p_obs;
clearvars temp1 temp2 temp

tempTOP = beta .* alphaTOP;
tempBOT = beta .* alphaBOT;

% exp_count_q_c1_t1(i) = P(q_1=i, c_1=1| obs). This is Qx1
exp_count_q_c1_t1 = tempBOT(:,1);

% exp_count_q_c0_sumt(i) = sum_{t=2}^T P(q_t=i, c_t=0 | obs). This is Qx1
exp_count_q_c0_sumt = sum(tempTOP(:,2:T),2);

% exp_count_c_sumt(c) = sum_t=1^T P(c_t = c | obs). This is Cx1
exp_count_c_sumt = zeros(C,1);
exp_count_c_sumt(1) = sum(sum(tempTOP));
exp_count_c_sumt(2) = sum(sum(tempBOT));

% exp_count_q_t(i,t) = P(q_t = i | obs). This is QxT
exp_count_q_t=tempTOP+tempBOT;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% previous forward code
% for t=2:T
%     for i = 1:Q
%         alphaTOP(i,t) = obslik(i,t) * globalProb(i) * mixPref(1) * sum(alphaTOP(:,t-1) + alphaBOT(:,t-1));
%         alphaBOT(i,t) = obslik(i,t) * mixPref(2) * sum(transmat(:,i) .* alphaTOP(:,t-1) + alphaBOT(:,t-1));
%     end
% end
%fprintf(1,'alpha is ');
%[alphaTOP;alphaBOT]
% previous backward code
% temp1 = globalProb(:).*mixPref(1);
% for t=T-1:-1:1
%     temp2 = beta(:,t+1) .* obslik(:,t+1);
%     for i=1:Q
%         temp3 = transmat(i,:)' .* mixPref(2);
%         beta(i,t) = sum( temp2 .* (temp1 + temp3 ));
%     end
% end
% previous code for psi 
% psi = zeros(Q,Q);
% temp1 = mixPref(2) .* (alphaTOP + alphaBOT);
% temp2 = beta .* obslik;
% for t=1:T-1
%     for i = 1:Q
%         for j = 1:Q
%             psi(i,j) = temp1(i,t) .* transmat(i,j) .* temp2(j,t+1);
%         end
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

