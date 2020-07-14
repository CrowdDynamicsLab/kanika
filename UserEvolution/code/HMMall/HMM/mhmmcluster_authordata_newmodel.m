O = 5; % 3(5) for stackexchange, 5 for MAG
M = 1;
K = 5; % Number of Clusters
iterations = 10;
cov_type = 'diag';
regCoeff = 0.001;

for Q = 4
    Likelihoods = [];
    num = (Q-1);
    D = Q+num;
    for attempts = 1:1
    

    clear prior*
    clear transmat*
    clear Sigma*
    clear mu*
    clear mixmat*

for i = 1:K
    prior0{i} = normalise(rand(Q,1));
    transmat0{i} = mk_stochastic(rand(Q,Q));
end
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
mu0 = {};
Sigma0 = {};
weightmat0 = {};
features = {};
% priormu = [0.99 0 0.01 0 0;
%            0 0 0 0.0 0;
%            0.0 0.97 0.01 0 0;
%            0.1 0.07 0.77 0.04 0.01;
%            0.52 0.42 0.03 0.01 0.01;
%            0.07 0.02 0.03 0.86 0.02];

    %% For stack exchange
% priormu = [0.96 0.01 0.02;
%            0 0 0;
%            0.03 0.91 0.05;
%            0.04 0.06 0.89;
%            0.03 0.42 0.54;
%            0.4 0.07 0.52;
%            0.03 0.25 0.71];

priormu = [0.96 0.01 0.02;
           0 0 0;
           0.03 0.91 0.05;
           0.04 0.06 0.89];

for i = 1:K
 [mu0{i}, Sigma0{i}] = mixgauss_init(Q*M, cat(2,author_data_cell{:}), cov_type);
  mu0{i} = reshape(mu0{i}, [O Q M]);
%Sigma0{i} = reshape(Sigma0{i}, [O O Q M]);



%mu0{i} = priormu';
Vector = 0.1;
Vector = repmat(Vector,O,1);
A = diag(Vector);

Sigma0{i} = repmat(A,1,1,Q);
%mixmat0 = mk_stochastic(rand(Q,M));
mixmat0 = ones(Q,1);
weightmat0{i} = rand(Q,D);

end
%% Algorithm
epsilon=1e-10;

% First iteration, find the data partition
loglik = zeros(K,1);
index = ones(K,1);
Cluster_Assign = zeros(size(author_data_cell,3),1);

%% Find initial features
for i = 1:K
    transmat0{i} = zeros(Q,Q);
    features{i} = zeros(Q,D); % could be Q * T * D
        for q = 1:Q
            features{i}(q,q)=1;
            k = 1;
            for j = 1:Q
                if(q ~= j)
                    %features{i}(q,Q+k) = distance_unranked(mu0{i},q,j);
                    features{i}(q,Q+k) = distance_diff(mu0{i},q,j);
			k = k+1;
                end
            end
        end


    for q = 1:Q     
        transmat0{i}(q,:) = (exp(weightmat0{i} * features{i}(q,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
    end
    transmat0{i} = mk_stochastic(transmat0{i});
end

%%

for j = 1:size(author_data_cell,3)   
    for i = 1:K
        
        loglik(i) = mhmm_logprob_newmodel(author_data_cell{j}, prior0{i},features{i},weightmat0{i}, mu0{i}, Sigma0{i});
    end
    [maxv,maxindex] = max(loglik);
    Cluster_Assign(j) = maxindex;
    
end
%Partition the data

%% Training Algorithm
for iter = 1:iterations
% Train the models seperately
    totalloglik = 0;
    for i = 1:K
        index = find(Cluster_Assign == i);
        author_data_cluster = {};
        for j = 1:size(index)
            author_data_cluster{j} = author_data_cell{index(j)};
            %author_data_cluster(:,:,j) = author_data_cell{index(j)};
        end
        transmat0{i} = zeros(Q,Q);
        features{i} = zeros(Q,D); % could be Q * T * D
        for q = 1:Q
            features{i}(q,q)=1;
            k = 1;
            for j = 1:Q
                if(q ~= j)
                    %features{i}(q,Q+k) = distance_unranked(mu0{i},q,j);
		    features{i}(q,Q+k) = distance_diff(mu0{i},q,j);
                    k = k+1;
                end
            end
        end


    for q = 1:Q     
        transmat0{i}(q,:) = (exp(weightmat0{i} * features{i}(q,:)'))'; %computing w1*F1 w2*F1 w3*F1 ...
    end
    transmat0{i} = mk_stochastic(transmat0{i});
        
        [LL, prior0{i}, transmat0{i}, weightmat0{i}, mu0{i}, Sigma0{i}] = ...
        mhmm_em_newmodel(author_data_cluster, prior0{i}, transmat0{i}, weightmat0{i}, features{i},mu0{i}, Sigma0{i}, regCoeff,'thresh',epsilon,'verbose',0);
        totalloglik = totalloglik + LL(length(LL));
    end
    fprintf(1, 'After iteration %d, Totalloglik = %f\n', iter, totalloglik)
    
    %reassign data
    loglik = zeros(K,1);
    Cluster_Assign = zeros(size(author_data_cell,3),1);
    for j = 1:size(author_data_cell,3)
        for i = 1:K
            loglik(i) = mhmm_logprob_newmodel(author_data_cell{j}, prior0{i}, features{i}, weightmat0{i}, mu0{i}, Sigma0{i});
        end
        [maxv,maxindex] = max(loglik);
        Cluster_Assign(j) = maxindex;
    end
    
    
end

Likelihoods(attempts) = totalloglik;
path_cluster = zeros(size(author_data_cell,3),1);
for j = 1:size(author_data_cell,3)
    %B = mixgauss_prob(author_data(:,:,i), mu1, Sigma1, mixmat1);
    for i = 1:K
            loglik(i) = mhmm_logprob_newmodel(author_data_cell{j}, prior0{i}, features{i}, weightmat0{i}, mu0{i}, Sigma0{i});
    end
    [maxv,maxindex] = max(loglik);    
    B = mixgauss_prob(author_data_cell{j}, mu0{maxindex}, Sigma0{maxindex}, mixmat0);
    path{j} = viterbi_path(prior0{maxindex}, transmat0{maxindex}, B);
    path_cluster(j) = maxindex;
end

    end
end
