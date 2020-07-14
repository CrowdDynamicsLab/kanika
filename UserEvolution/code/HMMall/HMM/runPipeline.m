tic
addpath(genpath('/home/kanika/HMMall/'))
%load s
%load '../StackExchangeData/sessiondata_10_filtered750.mat'
%load '../MicrosoftData/authordata_15_arealimited_v3_6dim_original_removepre0s.mat'
%load '../MicrosoftData/authordata_15_arealimited_hari_thresh3and3_active15.mat'
%load '../StackExchangeData/sessiondataquantized_10.mat'
%author_data_cell=whole_author_data_cell(:,1:size(whole_author_data_cell,3));
%author_data_cell=whole_author_data_cell;
size(author_data_cell)

%myCluster = parcluster('MATLAB Parallel Cloud')
%myCluster=parcluster()
%myCluster.NumWorkers = 30
%saveProfile(myCluster)
%myCluster=parcluster()
%parcluster(myCluster,30)
%parpool('local',2)

priormu = [
	   0.99 0.0 0.001 0.0 0.0;
           0.36 0.01 0.54 0.08 0.0;
           0.08 0.01 0.04 0.76 0.0;
           0.01 0.1 0.98 0.01 0.0;
           0.01 0.4 0.48 0.1 0.02;
           0.0 0.97 0.0 0.0 0.0
];

%priormu = [0.99 0.0 0.001 0.0 0.1;
%           0.16 0.01 0.84 0.08 0.1;
%           0.08 0.01 0.04 0.76 0.1;
%           0.0 0.97 0.0 0.0 0.1];

%For author data
%priormu = [0.92 0.02 0.01 0.01 0.01;
%	   0.01 0.02 0.02 0.01 0.01;
%	   0.47 0.27 0.02 0.01 0.01;
%	   0.08 0.82 0.01 0.07 0.04;
%           0.07 0.11 0.66 0.04 0.03;
	   %0.08 0.08 0.07 0.48 0.23
%	  ];

%priormu = [
           %0.0001 0.001 0.0007 0.0003 0.009;
%	   0.98   0.01  0.002  0.003  0.002;
%           0.01 0.01  0.07   0.03 0.09;
%           0.47 0.46  0.035  0.02   0.012;
%           %0.47   0.01  0.29   0.07   0.06;
%	   0.02   0.91  0.04   0.02   0.01;
  %         0.03   0.13  0.77   0.05   0.03;
           %0.11   0.13  0.01   0.59   0.07
 %         ];

priormu = [
        %   0.0001 0.001 0.0007 0.0003 0.009;
          0.02 0.96  0.008  0.003  0.001  0.0004;
          0.86 0.08 0.03  0.01  0.005   0.004;
	   %0.32 0.53 0.08  0.003   0.02 0.009;
          0.13 0.46 0.27 0.08 0.04 0.02; 
          0.18 0.13   0.61  0.04   0.02   0.01;
         % 0.18 0.13   0.15  0.47   0.04   0.02;
         0.11 0.07   0.07  0.11   0.59   0.12
         ];

cluster_log = [];
for C = 4:4
[prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(author_data_cell,s,priormu,'states',5,'clusters',C,'observations',6, 'muop','prior');

%[prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
%    mhmm_authordata_individual_withlog(author_data_cell,s,priormu,'states',5,'clusters',C,'observations',6);
%[prior1,transmat1,obsmat1,loglik,path] = dhmm_authordata(author_data_cell,s);

%mu0{1}
%mu0{2}
%mu0{3}
%mu0{4}
%ClusterQualityPaper
%mhmm_authordata
%dhmmcluster_authordata
%mhmmcluster_authordata
'Model Computed', C, totalloglik
cluster_log(C-1)=totalloglik/size(author_data_cell,2);
end
toc
cleaner = onCleanup(@() delete(gcp('nocreate')));
%save('../StackExchangeData/resultsession_mu_Q6K4_10_filtered750.mat', mu0)
%save '../StackExchangeData/resultsessionclustergaussQ5K4_10_filtered750.mat'
%save '../MicrosoftData/resultauthorclustergaussK4_thresh3and3_active15_GCluster.mat'
%PlotHMM(4,obsmat1,transmat1)
%PlotHMM(5,mu0,transmat0)
%% Clustering
%AnalyseStateDecoding
%ClusterData
