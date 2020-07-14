%clear
%load ../MicrosoftData/resultauthorclustergaussQ5K7_15_arealimited_v3_remove0s.mat
%load ../MicrosoftData/resultauthorclustergaussQ5K4_15_arealimited_hari_thresh3and3_active15.mat
%load ../StackExchangeData/resultsessionclustergaussQ5K4_10_filtered750.mat
%loglikmatrix
%clusterassign

%% Find inter-cluster distance metric.
K = size(unique(path_cluster),1);
ClusterSim = zeros(K,K);
%% Find KL divergence between most probable path and states assigned
for k = 1:K %All clusters
    for l = k+1:K
        index1 = find(path_cluster==k);
        index2 = find(path_cluster==l);
        %index = [index1; index2];
        d1 = sum(loglikmatrix(index1,k)-loglikmatrix(index1,l))/length(index1);
        d2 = sum(loglikmatrix(index2,l)-loglikmatrix(index2,k))/length(index2);
        ClusterSim(k,l) = 0.5 *(d1+d2);
    end
end

ClusterSim = ClusterSim + ClusterSim';


%% Find intra-cluster distance
ClusterIntra = zeros(K,1);
for k = 1:K
    index = find(path_cluster==k);
    N = length(index);
    %temp = zeros((N*(N-1)),1);
    %temp = zeros(100000, 1);
    values = loglikmatrix(index,k);
    %for i = 1:100000
    %	indice = randi([1 N],1,2);
        %start = N*(i-1);
        %temp(start+1:start+N) = values-values(i);
    %	temp(i) = values(indice(1)) - values(indice(2));
    % end
    %ClusterIntra(k,1) = sum(loglikmatrix(index,k))/length(index);
    %ClusterIntra(k,1) = var(temp);
    ClusterIntra(k,1) = mad(values);
end

'Intra-cluster distance is', ClusterIntra
K
ClusterSim
ClusterIntra
dunn_index = min(ClusterSim(ClusterSim>0))/max(ClusterIntra)
db_index = -1000000000.0;
for i = 1:K
	for j =i+1:K
		ratio = (ClusterIntra(i) + ClusterIntra(j))/ClusterSim(i,j);
		if ratio > db_index
			db_index = ratio;
		end
	end
end

db_index
