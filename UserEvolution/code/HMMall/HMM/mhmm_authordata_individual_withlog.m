%% Implementing baseline cluster HMM
function [prior,transmat,mu,Sigma,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmm_authordata_individual_withlog(author_data_cell,s,priormu,varargin)


%load s
%Likelihoods = [];

[states, nobserv, ncluster, priorop, transmatop, muop] = ...
   process_options(varargin, 'states', 6,'observations',5 ,'clusters',4, 'prior', 'equal',...
                                'transmat','fullLR','muop','prior');

datasize = size(author_data_cell,2);
DistMatrix = zeros(datasize,datasize,'uint8');
C=ncluster;              

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
    transmat0 =mk_stochastic(transmat0);
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
    mu0=mk_stochastic(rand(O,Q));
else
     A = repmat((1/O),O,1);
     mu0 = repmat(A,1,Q);
end

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

parfor l = 1: datasize   
 %fprintf(1,'Author %d\n',l);
 [LL, prior1, transmat1, mu1, Sigma1] = ...
    mhmm_em(author_data_cell{l}, prior0, transmat0, mu0, Sigma0, 'thresh',epsilon,'verbose',0,'adj_Sigma',0);
 
logprob = zeros(datasize,1);
for j = 1:size(author_data_cell,2)
    logprob(j,1) = mhmm_logprob(author_data_cell{j}, prior1, transmat1, mu1, Sigma1);
end
DistMatrix(:,l) = logprob;
end

%% Find clusters
Tempdist = zeros(datasize,datasize);
for i = 1:datasize
    for j = i+1:datasize
        Tempdist(i,j)=0.5*(DistMatrix(i,i) - DistMatrix(i,j) + DistMatrix(j,j) - DistMatrix(j,i));
    end
end

Tempdist = Tempdist + Tempdist';
%DistMatrix = Tempdist;
%clear Tempdist;
[path_cluster,cluster] = kmedioids(Tempdist,C);
'Computed'

%% Find cluster HMMs
for i = 1:C
     [LL, prior{i}, transmat{i}, mu{i}, Sigma{i}] = ...
        mhmm_em(author_data_cell{cluster(i)}, prior0, transmat0, mu0, Sigma0,'thresh',epsilon,'verbose',0,'adj_Sigma',0);
end

%% Find loglik matrix
loglikmatrix = zeros(datasize,C);
totalloglik = 0.0;
for i = 1:C
    loglikmatrix(:,i) = DistMatrix(:,i);
end

totalloglik = sum(sum(loglikmatrix));

%% Find paths
for j = 1:size(author_data_cell,2)
    maxindex = path_cluster(j);
    B = mixgauss_prob(author_data_cell{j}, mu{maxindex}, Sigma{maxindex}, ones(Q,1));
    %B = mixgauss_prob(author_data_cell{j}, mu0, Sigma0, ones(Q,1));
    path{j} = viterbi_path(prior{maxindex}, transmat{maxindex}, B);
end
path_cluster=path_cluster';
