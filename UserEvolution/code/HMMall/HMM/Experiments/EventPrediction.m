
%addpath(genpath('/home/kanika/HMMall/'))
% load ../StackExchangeData/sessiondata_10.mat
%load s
% 
 %Convert data into 90\% training and 10% testing
traindata = cellfun(@(x) x(:,1:int32(0.9*size(x,2))), author_data_cell, 'UniformOutput', false);
testdata = cellfun(@(x) x(:,int32(0.9*size(x,2))+1:end), author_data_cell, 'UniformOutput', false);
% 
%load ../StackExchangeData/sessiondata_10_filtered750_traintest.mat
%load ../MicrosoftData/authordata_15_arealimited_hari_thresh3and3_new_traintest.mat
%load ../MicrosoftData/authordata_15_arealimited_hari_thresh3and3_active15_traintest.mat
size(traindata)
%parpool('local',2)
tic
priormu = [
           0.99 0.0 0.001 0.0 0.0;
           0.36 0.01 0.54 0.08 0.0;
           0.08 0.01 0.04 0.76 0.0;
           0.01 0.1 0.98 0.01 0.0;
          0.01 0.4 0.48 0.1 0.02;
           0.01 0.97 0.01 0.01 0.01;
           ];

% priormu = [
%            %0.0001 0.001 0.0007 0.0003 0.009;
%            0.01 0.98   0.01  0.002  0.003  0.002;
%            0.99 0.0001 0.001 0.0007 0.0003 0.009;
%            0.001 0.47   0.46  0.035  0.02   0.012;
%            %0.47   0.01  0.29   0.07   0.06;
%            0.001 0.02   0.91  0.04   0.02   0.01;
%            0.001 0.03   0.13  0.77   0.05   0.03;
%            %0.11   0.13  0.01   0.59   0.07
%           ];       
% 
% priormu = [
%            %0.0001 0.001 0.0007 0.0003 0.009;
%            0.02 0.96  0.008  0.003  0.001  0.0004;
%           0.86 0.08 0.03  0.01  0.005   0.004;
% %           0.32 0.53 0.08  0.003   0.02 0.009;
% %           %0.47   0.01  0.29   0.07   0.06;
%            0.13 0.46 0.27 0.08 0.04 0.02;
%            0.18 0.13   0.61  0.04   0.02   0.01;
%            0.18 0.13   0.15  0.47   0.04   0.02;
% %           0.11 0.07   0.07  0.11   0.59   0.12
%           ];

%[prior1,transmat1,mu1,Sigma1,LL,path] = mhmm_authordata(traindata,s,priormu,'states',5,'observations',6,...
%                                'prior', 'equal','transmat','fullLR','muop','prior');
                                                                       
[prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(traindata,s,priormu,'states', 6,'observations',5, 'clusters',4,'prior', 'equal',...
                                'transmat','fullLR','muop','prior');

toc
%% With mhmm_authordata
% loglik = 0.0;
% if((exist ('path_cluster')) == 1)
% 
%     for i = 1:size(testdata,2)
%         data = testdata{i};
%         N = size(testdata{i},2);
%         Q = path{1,i}(end);
%         K = path_cluster(i);
%         loglik = loglik + sum(gaussian_prob(data, mu0{K}(:,Q), Sigma0{K}(:,:,Q), 1)/N);
%     
%     end
%     loglik = loglik/size(testdata,2);
% 
% 
% else
% 
%     for i = 1:size(testdata,2)
%         data = testdata{i};
%         N = size(testdata{i},2);
%         Q = path{1,i}(end);
%         loglik = loglik + sum(gaussian_prob(data, mu1(:,Q), Sigma1(:,:,Q), 1)/N);
%     
%     end
%     loglik = loglik/size(testdata,2);
% 
% end

%save '../StackExchangeData/EventpredMostProbPathStackSessionClus_10.mat'
%load '../StackExchangeData/EventpredMostProbPathStackSessionClus_10.mat'
tic

%%

% normalize each mu
mu1 = mu0;
for s = 1:size(mu0,2)
    for r = 1:size(mu0{1,s},2)
        mu1{1,s}(:,r) = mu0{1,s}(:,r)/sum(mu0{1,s}(:,r));
    end
end

jsdiv = 0.0;
totalnelem = 0;
if((exist ('path_cluster')) == 1)
    
    
    parfor i = 1:size(testdata,2)
        %i
        K = path_cluster(i);
        Q = size(prior0{K},1);
       %B = mixgauss_prob(whole_author_data_cell{i}, mu0{K}, Sigma0{K}, ones(Q,1));
        B = mixgauss_prob(testdata{i}, mu1{K}, Sigma0{K}, ones(Q,1));
        path = viterbi_path(prior0{K}, transmat0{K}, B);
        
        for p = 1:size(path,2)
            if sum(testdata{i}(:,p)) == 0
                continue
            end
            
            %R = mvnrnd(mu1{K}(:,path(p)),Sigma0{K}(:,:, path(p)),1);
            %normalize R
            %R(R < 0) = 0; R = R/norm(R,1);
            %jsdiv = jsdiv + KLmultinomial(testdata{i}(:,p)', R, 'js');
            
            jsdiv = jsdiv + KLmultinomial(testdata{i}(:,p), mu1{K}(:,path(p)), 'js');
            
            totalnelem = totalnelem + 1;
        end
        %R = mvnrnd(mu0{K}(:,n),Sigma0{K}(:,n),1);
        
	%totalnelem = totalnelem + size(path,2);
    
    end
    %loglik = loglik/size(testdata,2);
     jsdiv = jsdiv/totalnelem


else

    parfor i = 1:size(testdata,2)
        Q = size(prior1,1);
        %B = mixgauss_prob(whole_author_data_cell{i}, mu1, Sigma1, ones(Q,1));
        B = mixgauss_prob(testdata{i}, mu1, Sigma1, ones(Q,1));
        path = viterbi_path(prior1, transmat1, B);
        for p = 1:size(path,2)
            if sum(testdata{i}(:,p)) == 0
                continue
            end
            jsdiv = jsdiv + KLmultinomial(testdata{i}(:,p), mu1(:,path(p)), 'js');
            totalnelem = totalnelem + 1;
        end
    
    end
    jsdiv = jsdiv/totalnelem

end

'Jenson Shannon divergence is',jsdiv



%% Only the testing data probability
% loglik = 0.0;
% totalnelem = 0;
% if((exist ('path_cluster')) == 1)
%     
%     
%     parfor i = 1:size(testdata,2)
%         N = size(traindata{i},2);
%         K = path_cluster(i);
%         Q = size(prior0{K},1);
%        %B = mixgauss_prob(whole_author_data_cell{i}, mu0{K}, Sigma0{K}, ones(Q,1));
%         B = mixgauss_prob(author_data_cell{i}, mu0{K}, Sigma0{K}, ones(Q,1));
%         R = mvnrnd(mu0{K},Sigma0{K},1)
%         [lik, path,nelem] = viterbi_path_testdata(prior0{K}, transmat0{K}, B,N);
%         loglik = loglik + lik; %(Normalized value)
% 	totalnelem = totalnelem + nelem;
%     
%     end
%     %loglik = loglik/size(testdata,2);
%      loglik = loglik/totalnelem
% 
% 
% else
% 
%     parfor i = 1:size(testdata,2)
%         N = size(traindata{i},2);
%         Q = size(prior1,1);
%         %B = mixgauss_prob(whole_author_data_cell{i}, mu1, Sigma1, ones(Q,1));
%         B = mixgauss_prob(author_data_cell{i}, mu1, Sigma1, ones(Q,1));
%         [lik, path,nelem] = viterbi_path_testdata(prior1, transmat1, B,N);
%         loglik = loglik + lik;
%         totalnelem =totalnelem + nelem;
%     
%     end
%     %loglik = loglik/size(testdata,2);
%     loglik = loglik/totalnelem
% 
% end
% 
% 'Loglikelihood is',loglik
toc

%cleaner = onCleanup(@() delete(gcp('nocreate')));
%save '../StackExchangeData/resultEventstacksessionclus_10.mat'
%save '../StackExchangeData/EventpredMostProbPathStackSession_10.mat'

%save '../MicrosoftData/EventpredMostProbPathauthor_15_v3_nocluster.mat'
