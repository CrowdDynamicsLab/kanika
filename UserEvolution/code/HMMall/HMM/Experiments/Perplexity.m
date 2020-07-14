%addpath(genpath('/home/kanika/HMMall/'))
load s;
%%Create train and test data
%load ../StackExchangeData/sessiondata_10_filtered750.mat
%load '../MicrosoftData/authordata_15_arealimited_hari_thresh3and3_new.mat'
%load '../MicrosoftData/authordata_15_arealimited_hari_thresh3and3_active15.mat'
length = size(author_data_cell,2)
size(author_data_cell{1})
CV = 5; %Cross validation
K = 4; %Clusters
C = 6;
Q = 6; O = 5;
priormu = [
	   0.99 0.0 0.001 0.0 0.01;
           0.36 0.01 0.54 0.08 0.01;
           0.08 0.01 0.04 0.76 0.0;
           0.01 0.1 0.98 0.01 0.0;
           0.01 0.4 0.48 0.1 0.02;
           0.0 0.97 0.0 0.0 0.01
];

%priormu = [
           %0.0001 0.001 0.0007 0.0003 0.009;
%          0.01 0.98   0.01  0.002  0.003  0.002;
%          0.99 0.0001 0.001 0.0007 0.0003 0.009;
      %    0.01 0.47   0.46  0.035  0.02   0.012;
%          0.47   0.01  0.29   0.07   0.06;
%          0.01 0.02   0.91  0.04   0.02   0.01;
%          0.01 0.03   0.13  0.77   0.05   0.03;
           %0.11   0.13  0.01   0.59   0.07
%          ];

%%priormu = [
           %0.0001 0.001 0.0007 0.0003 0.009;
%%           0.02 0.96  0.008  0.003  0.001  0.0004;
%%          0.86 0.08 0.03  0.01  0.005   0.004;
%%           0.32 0.53 0.08  0.003   0.02 0.009;
%           %0.47   0.01  0.29   0.07   0.06;
%%           0.13 0.46 0.27 0.08 0.04 0.02; 
%%           0.18 0.13   0.61  0.04   0.02   0.01;
%           0.18 0.13   0.15  0.47   0.04   0.02;
%           0.11 0.07   0.07  0.11   0.59   0.12
%%          ]; 
	

size(priormu);
%parpool('local',2)
%Start rotating
Perplex = zeros(CV,1);
Testlog = zeros(CV,1);
for K = C:C
indices = crossvalind('Kfold',length,CV);
for i = 1:CV
    i
    test = (indices == i); train = ~test;
    traindata = author_data_cell(train);
    testdata = author_data_cell(test);
    %tic
    %traindata{1}
    [prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(traindata,s,priormu,'states', Q,'observations',O, 'clusters',K,'prior', 'equal',...
                                'transmat','full','muop','prior');
    size(traindata)
    %toc
    %traindata{1}    
    %[prior0,transmat0,mu0,Sigma0,totalloglik,path] = ...                        
    %mhmm_authordata(traindata,s,priormu,'states', Q,'observations',O,'prior', 'equal',...
    %                            'transmat','fullLR','muop','prior'); 
    
    perplex = 0;
    totaltestlog = 0;
    'Testing data probability calculate'
    totaltestlog = zeros(size(testdata,2),1);
    parfor m = 1:size(testdata,2)
        loglik = zeros(1,K);
         for n = 1:K
             loglik(n) = mhmm_logprob(testdata{m}, prior0{n}, transmat0{n}, mu0{n}, Sigma0{n});
             %loglik(n) = mhmm_logprob(testdata{m}, prior0, transmat0, mu0, Sigma0);
              %mhmm_logprob(testauthor_data_cell{j}, prior0{i}, transmat0{i}, mu0{i}, Sigma0{i});
         end
	 %loglik = vpa(normr(exp(loglik)))
         %loglik = log(loglik) %Normalize the output
         [maxv,maxindex] = max(loglik);
         %perplex = perplex + exp(-(maxv/(K*(Q*O + Q*Q + Q))));
         %perplex = perplex + -(maxv);
	 totaltestlog(m)=maxv;
    end
    i
    %Perplex(i)=exp(perplex/size(testdata,2));
    Testlog(i) = -sum(totaltestlog)
end

%Perplex,mean(Perplex)
Testlog,mean(Testlog)
PerplexK = (mean(Testlog)/size(testdata,2));
end

'Different perplexity for C'
PerplexK

%cleaner = onCleanup(@() delete(gcp('nocreate')));
%save '../MicrosoftData/resultauthordata_15_perplexity_nocluster.mat'
