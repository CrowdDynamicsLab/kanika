priormu = [
        %   0.0001 0.001 0.0007 0.0003 0.009;
          0.02 0.96  0.008  0.003  0.001  0.0004;
          0.86 0.08 0.03  0.01  0.005   0.004;
	   %0.32 0.53 0.08  0.003   0.02 0.009;
          0.13 0.46 0.27 0.08 0.04 0.02; 
          0.18 0.13   0.61  0.04   0.02   0.01;
         % 0.18 0.13   0.15  0.47   0.04   0.02;
         0.11 0.07   0.07  0.11   0.59   0.12;
         ];

distance = zeros(8,1);

for C = 2:10
    [prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(author_data_cell,s,priormu,'states',5,'clusters',C,'observations',6, 'muop','prior');

    %distance(C) = ClusterDiff(originalmu, mu0);
    distance(C) = totalloglik;
    
end
distance
mean(distance)
var(distance)
    
    