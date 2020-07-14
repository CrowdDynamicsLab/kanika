
'In normal Current State features'
Likelihoods = [];
for Q = 4
O = 4;
M = 1;
D = Q; %dimensionality of feature vector (currently same as Q)
%Q = 9;
%T=20;
%nex = 14;
cov_type = 'diag';

clear prior*
clear transmat*
clear Sigma*
clear mu*
clear mixmat*

regCoeff = 0.01;
prior0 = normalise(rand(Q,1));
weightmat0 = rand(Q,D);
%weightmat0 = lastfinalweight;

transmat0 = zeros(Q,Q);
features = zeros(Q,D); % could be Q * T * D
for i = 1:Q
    features(i,i)=1; %Putting the state ON
    transmat0(i,:) = (exp(weightmat0 * features(i,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
end

transmat0 = mk_stochastic(transmat0);
%transmat0 = mk_stochastic(rand(Q,Q));
% for i = 1: size(transmat0,1)
%     for j = 1:i-1
%         transmat0(i,j) = 0;
%     end
% end
%% other way of intializing
% Sigma0 = repmat(eye(O), [1 1 Q M]);
%   % Initialize each mean to a random data point
% indices = randperm(T*nex);
% mu0 = reshape(author_data(:,indices(1:(Q*M))), [O Q M]);
% mixmat0 = mk_stochastic(rand(Q,M));

  %% initial guess of parameters

priormu = [0.99 0 0.01 0;
           0 0.9 0 0.0;
           0.0 0.1 0.9 0 ;
           0 0.1 0. 0.9];
       
priormu = [0.99 0 0.01 0 0;
           0 0 0 0.0 0;
           0.0 0.97 0.01 0 0;
           0.1 0.07 0.77 0.04 0.01;
           0.52 0.42 0.03 0.01 0.01;
           0.07 0.02 0.03 0.86 0.02];
       
%[mu0, Sigma0] = mixgauss_init(Q*M, cat(2,author_data_cell{:}), cov_type);
%mu0 = reshape(mu0, [O Q M]);
%Sigma0 = reshape(Sigma0, [O O Q M]);

mu0 = priormu';
Vector = 0.1;
Vector = repmat(Vector,O,1);
A = diag(Vector);

Sigma0 = repmat(A,1,1,Q);
%mixmat0 = mk_stochastic(rand(Q,M));
mixmat0 = ones(Q,1);

%% Algorithm
epsilon=1e-5;
 
[LL, prior1, transmat1, weightmat1, mu1, Sigma1, mixmat1] = ...
   mhmm_em_newmodel(author_data_cell, prior0, transmat0, weightmat0, features, mu0, Sigma0, mixmat0, regCoeff, 'thresh',epsilon,'adj_mix',0,'verbose',1);
loglik = mhmm_logprob(author_data_cell, prior1, transmat1, mu1, Sigma1, mixmat1);  % To change


Likelihoods(Q) = loglik;
for i = 1:size(author_data_cell,3)
    %B = mixgauss_prob(author_data(:,:,i), mu1, Sigma1, mixmat1);
    B = mixgauss_prob(author_data_cell{i}, mu1, Sigma1, mixmat1);
    path(i,:) = viterbi_path(prior1, transmat1, B); %to change
end

end