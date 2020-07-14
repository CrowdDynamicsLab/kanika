
function [prior1,transmat1,mu1,Sigma1,LL,path] = ...
    mhmm_authordata(author_data_cell,s,priormu,varargin)

load s
%Likelihoods = [];

[states, nobserv, priorop, transmatop, muop] = ...
   process_options(varargin, 'states', 6,'observations',5, 'prior', 'equal',...
                                'transmat','fullLR','muop','prior');
for Q = states
O = nobserv;

if(strcmp(priorop,'rand'))
    rng(s);
    prior0 = normalise(rand(Q,1));
else
    prior0 = repmat((1/Q),Q,1);
end

if(strcmp(transmatop,'fullLR'))
    rng(s);
    transmat0 = mk_stochastic(rand(Q,Q));
    for i = 1: size(transmat0,1)
        for j = 1:i-1
            transmat0(i,j) = 0;
        end
    end
    transmat0=mk_stochastic(transmat0);
elseif(strcmp(transmatop,'halfLR'))
    transmat0 = mk_leftright_transmat(10,0.7);
else
    rng(s);
    transmat0 = mk_stochastic(rand(Q,Q));
end

if(strcmp(muop,'prior'))
    mu0 = priormu';
elseif(strcmp(muop,'rand'))
    rng(s);
    mu0{i}=mk_stochastic(rand(O,Q));
else
     A = repmat((1/O),O,1);
     mu0 = repmat(A,1,Q);
end

  %% initial guess of parameters (FOR DBLP DATA)
% priormu = [0.99 0 0.01 0 0;
%            0 0 0 0.0 0;
%            0.0 0.97 0.01 0 0;
%            0.1 0.07 0.77 0.04 0.01;
%            0.52 0.42 0.03 0.01 0.01;
%            0.07 0.02 0.03 0.86 0.02];
% 
% priormu = [0.99 0 0.01 0;
%            0 0.9 0 0.0;
%            0.0 0.1 0.9 0 ;
%            0 0.1 0. 0.9];
%
%% For STACK EXCHANGE DATA

% priormu = [0.96 0.01 0.02;
%            0 0 0;
%            0.03 0.91 0.05;
%            0.04 0.06 0.89;
%            0.03 0.42 0.54;
%            0.4 0.07 0.52;
%            0.03 0.25 0.71];
       
% priormu = [0.36 0.01 0.54 0.08 0.0;
%            0.08 0.1 0.04 0.76 0.0;
%            0.01 0.28 0.48 0.19 0.02;
%            0.99 0.0 0.0 0.0 0.0;
%            0.0 0.0 0.97 0.0 0.0;
%            0.01 0.1 0.95 0.01 0.0];

% priormu = [0.99 0.0 0.0 0.0 0.0;
%            0.36 0.01 0.54 0.08 0.0;
%            0.08 0.1 0.04 0.76 0.0;
%            0.01 0.1 0.98 0.01 0.0;
%            0.01 0.4 0.48 0.1 0.02;
%            0.0 0.97 0.0 0.0 0.0];

%% According to cluster infor

% priormu = [0.99 0.0 0.001 0.0 0.0;
%            0.36 0.01 0.54 0.08 0.0;
%            0.08 0.01 0.04 0.76 0.0;
%            0.01 0.1 0.98 0.01 0.0;
%            0.01 0.4 0.48 0.1 0.02;
%            0.0 0.97 0.0 0.0 0.0];
% 
%  mu0 = priormu';

%[mu0, Sigma0] = mixgauss_init(Q*M, cat(2,author_data_cell{:}), cov_type);
%mu0 = reshape(mu0, [O Q M]);
 %Sigma0 = reshape(Sigma0, [O O Q M]);

Vector = 0.01;
Vector = repmat(Vector,O,1);
A = diag(Vector);

Sigma0 = repmat(A,1,1,Q);
%mixmat0 = mk_stochastic(rand(Q,M));
mixmat0 = ones(Q,1);

%% Algorithm
epsilon=1e-4;
 
 tic
 [LL, prior1, transmat1, mu1, Sigma1] = ...
    mhmm_em(author_data_cell, prior0, transmat0, mu0, Sigma0, 'thresh',epsilon,'verbose',1,'adj_Sigma',0);
 toc
%testauthor_data_cell=whole_author_data_cell(:,5001:5500);
%loglik = mhmm_logprob(testauthor_data_cell, prior1, transmat1, mu1, Sigma1);
%perplexity = exp(-(loglik/(Q*O + Q*Q + O*O*Q + Q)))


%Likelihoods(Q) = LL;
path = cell(1,size(author_data_cell,2));
%%Changed this
for i = 1:size(author_data_cell,2)
    %B = mixgauss_prob(author_data(:,:,i), mu1, Sigma1, mixmat1);
    B = mixgauss_prob(author_data_cell{i}, mu1, Sigma1, mixmat0);
    path{i} = viterbi_path(prior1, transmat1, B);
end

end