%Assume have been given path and cluster assignments

%load ../StackExchangeData/resultsession4Q5K.mat
load '../StackExchangeData/resultsessionclustergauss_10.mat'

%path= path(1:5000);
%Cluster_Assign = Cluster_Assign(1:5000);

K = size(unique(Cluster_Assign),1); % or get as input
Q = size(prior0{1},1);

tic
ClusterSim = zeros(K,Q);
%% Find KL divergence between most probable path and states assigned
for k = 1:K %All clusters
    index = find(Cluster_Assign==k);
    Clusterdata = path(index);
    ClusterOrgdata = author_data_cell(index);
    StageData = {};
    for q = 1:Q
        index2 = cellfun(@(x)find(x==q),Clusterdata,'UniformOutput',false);
        data = cellfun(@(x,y)x(:,y),ClusterOrgdata,index2,'UniformOutput',false);
        StageData{q} = cell2mat(data)';
        
    end
%     for i =1:size(Clusterdata,2)
%         seq = Clusterdata{i};
%         for q = 1:Q
%             index2 = find(seq==q);
%             data = author_data_cell{index(i)}(:,index2);
%             StageData{q} = [StageData{q},data'];
%         end            
%     end
    
    %Calculate pairwise stagewise similarity
%     for q = 1:Q
%         [row,col] = size(StageData{q});
%         dist = 0.0;
%         for r = 1:row
%             for r1 = r+1:row
%                 dist = dist + KLmultinomial(StageData{q}(r,:),StageData{q}(r1,:));
%             end
%         end
%         ClusterSim(k,q) = dist/sum(1:r-1);
%     end
    
%% Calculate distance from the centroid
    for q = 1:Q
        [row,col] = size(StageData{1,q});
        dist = 0.0;
        mu_cluster = mu0{1,k}(:,q);
        for r = 1:row
            dist = dist + KLmultinomial(StageData{1,q}(r,:)',mu_cluster,'js');
        end
        ClusterSim(k,q) = dist/row;
    end
end
     


toc
ClusterSim
Totalsim = sum(sum(ClusterSim,2))/K;

fprintf('Total intra cluster similarity is(combined Q and not) %f, %f\n',Totalsim,Totalsim/Q);

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
ClusterDist_stage = cell(K,K);
for k = 1:K
    for l = k:K
        div = 0.0;
        for q = 1:Q
            %div = div + KLmultinomial(mu0{1,k}(:,q),mu0{1,l}(:,q),'js');
            val = KL(mu0{1,k}(:,q),mu0{1,l}(:,q),A);
            div = div +val ;
            ClusterDist_stage{k,l} = [ClusterDist_stage{k,l},val];
        end
        ClusterDist(k,l) = div;
    end
end
%ClusterDist = ClusterDist + ClusterDist';
fprintf('Total inter-cluster distance is %f\n',sum(sum(ClusterDist))/sum(1:(K-1)));

save '../StackExchangeData/resultsessionclustergauss_10_KL.mat'
