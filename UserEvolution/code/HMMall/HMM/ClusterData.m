% [idX,C,sumd] = kmeans(cell2mat(DummyStringData),2,'Replicates',5);
% %[mem cent] = ksc_toy(DummyStringData,2);
% silhouette(path,mem)

%# instances

str = {};
for i = 1:size(path_Clusterone)
    str{i} = num2str(path_Clusterone(i,:));
end
numStr = numel(str);

% %# create and fill upper half only of distance matrix
D = zeros(numStr,numStr);
for i=1:numStr
    for j=i+1:numStr
        D(i,j) = levenshtein_distance(str{i},str{j});
    end
end
Dupper = D;
D = D + D';       %'# symmetric distance matrix
Dmat = D;
% %# linkage expects the output format to match that of pdist,
% %# so we convert D to a row vector (lower/upper part of matrix)

D = squareform(D, 'tovector');

T = linkage(D, 'complete');

Silhoutteval = [];
for i = 4
Ti = cluster(T,'maxclus',i);
silvalue = silhouette([],Ti,D);
savg = grpstats(silvalue,Ti);
Silhoutteval(i) = mean(savg);
end

 % dendrogram(T)

%% Find representative cluster center

% Put all objects belonging to each cluster in seperate places
Clusters = {}; %zeros(size(unique(Ti)));
for i = 1:range(size(unique(Ti)))+1
    Clusters{i} = find(Ti == i);
end
    
ClusterCenter = {};
for i = 1: range(size(Clusters))+1
    temp = Dmat(Clusters{i},Clusters{i});
    [sumv,index] = min(sum(temp,2));
    ClusterCenter{i} = str{Clusters{i}(index)};   
end

ClusterCenter = ClusterCenter';

for i = 1:range(size(ClusterCenter))+1
    ClusterCenter{i} = str2num(ClusterCenter{i});    
end

ClusterCenter = cell2mat(ClusterCenter);