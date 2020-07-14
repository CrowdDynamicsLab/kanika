%Assume have been given path and cluster assignments

% load ../StackExchangeData/resultsession4Q5K.mat

%path= path(1:5000);
%Cluster_Assign = Cluster_Assign(1:5000);

K = size(unique(Cluster_Assign),1); % or get as input

tic
%% Creating a compact length path 
compactpath={};
weightpath={};
samelencompactpath={};
for i =1:size(path,2)
    seq = path{i};
    temp = seq([1,diff(seq)]~=0);
    timestamp = find(diff([-1 seq -1]) ~= 0); % where does V change
    dif = diff(timestamp);
    temp=temp(dif>1); % retain only those symbols which are more than 1
    dif=dif(dif>1);
    %compactpath(i,1:size(temp,2))= temp;
    %weightpath(i,1:size(dif,2))=dif;
    compactpath{i}=temp;
    weightpath{i}=dif;
    
end
toc

tic
%% Calculate pairwise difference between items
ClusterSim = zeros(K);
%Distance={};
Medoids = {};
for k = 1:K
    index = find(Cluster_Assign==k);
    %D = zeros(size(index,1),size(index,1));
    D = zeros(size(index,1),1);
    ClusterSeq = compactpath(index);
    N = 0;
    sumc = 0;
    for i = 1:size(ClusterSeq,2)
            %fprintf('%d',i);
            D(i,1)= sum(cellfun(@(x) levenshtein_distance(x,ClusterSeq{i}),ClusterSeq),2);
            N=N+1;
    end
    fprintf('%d\n',k);
    %Distance{k}=D;
    ClusterSim(k)=sum(D)/(2*N);

    %Find medoid
    [val,index] = min(D);
    seq = find(Cluster_Assign==k);
    Medoids{k}=path{seq(index)};
end
toc

Totalsim = 0.0;
for i = 1:K
    Totalsim = Totalsim + ClusterSim(i)*(size(find(Cluster_Assign==k),1)/size(compactpath,2));
end

fprintf('Total intra cluster similarity is %f\n',Totalsim);

%% For inter cluster similarity

%Find medoid
%Medoids = {};
%for k = 1:K
%    [val,index] = min(sum(Distance{k},2));
%    seq = find(Cluster_Assign==k);
%    Medoids{k}=path{seq(index)};
%end

% Find distance between medoid and members of other cluster
ClusterDist = zeros(K,K);
for k = 1:K
    for l = k+1:K
        index = find(Cluster_Assign==k);
        ClusterSeq = compactpath(index);
        ClusterDist(k,l)=sum(cellfun(@(x) levenshtein_distance(x,Medoids{k}),ClusterSeq));
    end
end
ClusterDist = ClusterDist + ClusterDist';

fprintf('Total inter-cluster distance is %f\n',sum(sum(ClusterDist)));