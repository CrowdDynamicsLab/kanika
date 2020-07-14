function [prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(author_data_cell,s,priormu,varargin)

[states, nobserv, ncluster, priorop, transmatop, muop] = ...
   process_options(varargin, 'states', 6,'observations',5, 'clusters',4,'prior', 'equal',...
                                'transmatop','fullLR','muop','prior');

O = nobserv
M = 1;
K = ncluster % Number of Clusters
iterations = 100;
cov_type = 'diag';
thresh = 1e-4;
length = size(author_data_cell,2);
for Q = states
    Likelihoods = [];
for attempts = 1:1
    

for i = 1:K
    
    if(strcmp(priorop,'rand'))
        rng(s);
        prior0{i} = normalise(rand(Q,1));
    else
        prior0{i} = repmat((1/Q),Q,1);
    end
    
    if(strcmp(transmatop,'fullLR'))
        rng(s);
        transmat0{i} = mk_stochastic(rand(Q,Q));
        for k = 1: size(transmat0{i},1)
            for j = 1:k-1
                transmat0{i}(k,j) = 0;
            end
        end
        transmat0{i}=mk_stochastic(transmat0{i});
    elseif(strcmp(transmatop,'halfLR'))
        transmat0{i} = mk_leftright_transmat(10,0.7);
    else
        rng(s);
        transmat0{i} = mk_stochastic(rand(Q,Q));
    end
    
    if(strcmp(muop,'prior'))
        mu0{i} = priormu';
    elseif(strcmp(muop,'rand'))
        rng(s);
        mu0{i}=mk_stochastic(rand(O,Q));
    else
         A = repmat((1/O),O,1);
         mu0{i} = repmat(A,1,Q);
    end
end




  %% initial guess of parameters
%mu0 = {};

% priormu = [0.96 0.01 0.02;
%            0 0 0;
%            0.03 0.91 0.05;
%            0.04 0.06 0.89];

% priormu = [ 1.    0.    0.    0.    0.  ; 
%             0.    1.    0.    0.    0.;
%             0.    0.    1.    0.    0.;
%             0.    0.01  0.01  0.98  0.;
%             0.02  0.08  0.08  0.07  0.74;
%             0.5   0.    0.49  0.01  0.;
%              0.01  0.45  0.03  0.51  0.01;
%              0.09  0.02  0.44  0.44  0. ;
%             0.41  0.43  0.1   0.05  0. ;
%             0.45  0.    0.01  0.54  0. 
%             ];
% priormu = [0.99 0.0 0.0 0.0 0.0;
%            0.38 0.0 0.54 0.0 0.0;
%            0.0 0.1 0.04 0.76 0.0;
%            0.0 0.0 0.98 0.0 0.0;
%            0.0 0.4 0.48 0.1 0.02;
%            0.0 0.97 0.0 0.01 0.01];
       
priormu = [0.99 0.0 0.001 0.0 0.0;
           0.36 0.01 0.54 0.08 0.0;
           0.08 0.01 0.04 0.76 0.0;
           0.01 0.1 0.98 0.01 0.0;
           0.01 0.4 0.48 0.1 0.02;
           0.0 0.97 0.0 0.0 0.0];
%%
for i = 1:K

%[mu0{i}, Sigma0{i}] = mixgauss_init(Q*M, cat(2,author_data_cell{:}), cov_type);
%mu0{i} = reshape(mu0{i}, [O Q M]);
%Sigma0 = reshape(Sigma0, [O O Q M]);

Vector = 0.01;
Vector = repmat(Vector,O,1);
A = diag(Vector);

Sigma0{i} = repmat(A,1,1,Q);
%Sigma0 = repmat(A,1,1,Q);
end
%% Algorithm
epsilon=1e-4;

 loglik = zeros(K,1);
 index = ones(K,1);
 Cluster_Assign = zeros(size(author_data_cell,2),1);

%% First iteration, find the data partition 
% for j = 1:size(author_data_cell,3)   
%     for i = 1:K
%         loglik(i) = mhmm_logprob(author_data_cell{j}, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i}, mixmat0);
%     end
%     [maxv,maxindex] = max(loglik);
%     Cluster_Assign(j) = maxindex;
%     
% end

%%
llength = length/K;
for c = 1:K
    if(c ~= K)
        Cluster_Assign(int32((c-1)*llength)+1:int32(c*llength)) = c;
    else
         Cluster_Assign(int32((c-1)*llength)+1:end) = c;
    end
end

%% Partition the data

%% Training Algorithm
previous_loglik = -inf;
totalloglik = 0;
converged = 0;
num_iter = 1;
prevCluster_Assign = Cluster_Assign;
loglikmatrix = zeros(size(author_data_cell,2),K);
while (num_iter <= iterations) & ~converged
% Train the models seperately
    totalloglik = 0;
    tic
    for i = 1:K
        index = find(Cluster_Assign == i);
        author_data_cluster = author_data_cell(index);
 	%size(author_data_cluster)
        [LL, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i}] = ...
        mhmm_em(author_data_cluster, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i},'thresh',epsilon,'verbose',0,'adj_Sigma',0);
       % [LL, prior0{i}, transmat0{i}, mu0, Sigma0] = ...
       % mhmm_em(author_data_cluster, prior0{i}, transmat0{i}, mu0, Sigma0,'thresh',epsilon,'verbose',0,'adj_Sigma',0);
        totalloglik = totalloglik + LL(max(size(LL)));
    end
    toc
    fprintf(1, 'After iteration %d, Totalloglik = %f\n', num_iter, totalloglik)
    
    tic
    %reassign data
    loglik = zeros(K,1);
    prevCluster_Assign = Cluster_Assign;
    Cluster_Assign = zeros(size(author_data_cell,2),1);
    parfor j = 1:size(author_data_cell,2)
        loglik = zeros(K,1);
        for i = 1:K
            loglik(i) = mhmm_logprob(author_data_cell{j}, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i});
	    loglikmatrix(j,i)=loglik(i);
            %loglik(i) = mhmm_logprob(author_data_cell{j}, prior0{i}, transmat0{i}, mu0, Sigma0);
        end
	%loglik = normr(vpa(exp(loglik)))
	%loglik = log(loglik)
        [maxv,maxindex] = max(loglik);
        Cluster_Assign(j) = maxindex;
    end
    toc
    
    num_iter =  num_iter + 1;
    converged = em_converged(totalloglik, previous_loglik, thresh);
    previous_loglik = totalloglik;
    difference = length - sum(Cluster_Assign == prevCluster_Assign);
     if(difference <= (0.001*length))
         'Too less changes'
         break
     end
    
    
end

Likelihoods(attempts) = totalloglik;
path_cluster = zeros(size(author_data_cell,2),1);
for j = 1:size(author_data_cell,2)
    maxindex = Cluster_Assign(j);
    B = mixgauss_prob(author_data_cell{j}, mu0{maxindex}, Sigma0{maxindex}, ones(Q,1));
    %B = mixgauss_prob(author_data_cell{j}, mu0, Sigma0, ones(Q,1));
    path{j} = viterbi_path(prior0{maxindex}, transmat0{maxindex}, B);
    path_cluster(j) = maxindex;
end

%% Testing the likelihood
%     loglik = zeros(K,1);
%     totaltestlog = 0.0;
%     TestCluster_Assign = zeros(size(test_author_data_cell,2),1);
%     for j = 1:size(test_author_data_cell,2)
%         for i = 1:K
%             loglik(i) = mhmm_logprob(author_data_cell{j}, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i});
%             %loglik(i) = mhmm_logprob(test_author_data_cell{j}, prior0{i}, transmat0{i}, mu0, Sigma0);
%         end
%         [maxv,maxindex] = max(loglik);
%         TestCluster_Assign(j) = maxindex;
%         totaltestlog=totaltestlog+maxv;
%     end

    end
end
