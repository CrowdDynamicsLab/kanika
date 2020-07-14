function [] = PlotHMM(K,mu,transmat)
% for i = 1:size(mu0,2)
%     temp = mu0{1,i}';
%     save(feval'mu'+i+'.txt','temp','-ascii','-tabs');
%     temp = transmat0{1,i};
%     save('transmat'+i+'.txt','temp','-ascii','-tabs');
% end

% temp = mu1';
% save('mu1.txt','temp','-ascii','-tabs');
% temp = transmat1;
% save('transmat1.txt','temp','-ascii','-tabs');
% 
% D = size(features,2);
% for i = 1:Q
%     features(i,D) = 10/15;
% end
% 
% transmat1 = exp(weightmat1 * features')';
% transmat1 = mk_stochastic(transmat1);
% save('transmat10.txt','temp','-ascii','-tabs');

%K = 1; %CLuster size

% for k = 1:K
% temp = mu0{1,k}';
% save(strcat('mu',int2str(k),'.txt'),'temp','-ascii','-tabs');
% temp = transmat0{1,k};
% save(strcat('transmat',int2str(k),'.txt'),'temp','-ascii','-tabs');
% end

for k = 1:K
%temp = mu0';
if(K == 1)
    %mu = bsxfun(@minus,mu,mean(mu,2));
    %temp = mu';
    temp = mu; %% For discrete case
else
    temp = mu{k}';
end
%temp=obsmat1;
%temp=obsmat0{k};
save(strcat('../MatlabVariable/mu',int2str(k),'.txt'),'temp','-ascii','-tabs');
if(K == 1)
    temp = transmat;
else
    temp = transmat{k};
end
%temp=transmat0{k};
save(strcat('../MatlabVariable/transmat',int2str(k),'.txt'),'temp','-ascii','-tabs');
end