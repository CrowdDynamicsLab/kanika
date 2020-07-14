%save(strcat('../MatlabVariable/path.txt'),'path','-ascii','-tabs');
save(strcat('../MatlabVariable/ClusterAssign.txt'),'Cluster_Assign','-ascii','-tabs');

for i = 1:size(path,2)
    temp = path{i};
    save('../MatlabVariable/path.txt','temp','-ascii','-tabs','-append')
end
