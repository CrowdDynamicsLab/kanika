S = 4;
maxdistance = zeros(8,1);
mindistance = zeros(8,1);
for S = 3:3
    [prior0,transmat0,mu0,Sigma0,totalloglik,path,path_cluster,loglikmatrix] = ...
    mhmmcluster_authordata_withlog(author_data_cell,s,priormu,'states',S,'clusters',4,'observations',6, 'muop','rand');
    
%% Compute the KL divergence
    minVal = ones(1, size(mu0,2));
    maxVal = zeros(1, size(mu0,2));
    for c = 1:size(mu0,2)
        %for k = 1:size(mu0{1,1},2)
        for k = 1:S-1
            %for j = k+1:size(mu0{1,1},2)
            for j = S:S
                val = KLmultinomial(mu0{1,c}(:,k),mu0{1,c}(:,j),'js');
                %val = KL(mu0{1,c}(:,k),mu0{1,c}(:,j), Sigma0{1,1}(:,:,1));
                if val < minVal(1,c)
                    minVal(1,c) = val;
                end
                if val > maxVal(1,c)
                    maxVal(1,c) = val;
                end
            end
        end
    end

    maxdistance(S) = mean(maxVal);
    mindistance(S) = mean(minVal);
end