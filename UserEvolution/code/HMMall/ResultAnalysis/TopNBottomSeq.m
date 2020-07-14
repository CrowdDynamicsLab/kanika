K = 7 ; %Cluster size

TopPattern = {};
BottomPattern = {};
for i = 1:K
    index = 1;
    loglik = {};
for j = 1:size(author_data_cell,3)
    %B = mixgauss_prob(author_data(:,:,i), mu1, Sigma1, mixmat1);
    if(path_cluster(j) == i)
            loglik{index,1} = mhmm_logprob_newmodel(author_data_cell{j}, prior0{i}, features{i}, weightmat0{i}, mu0{i}, Sigma0{i})/size(path{j},2);
            loglik{index,2} = path{j};
            index = index + 1;
    end
end

   [trash idx] = sort([loglik{:,1}], 'descend'); 
   TopPattern{i,1} = loglik{idx(1),2};
   TopPattern{i,2} = loglik{idx(2),2};
   TopPattern{i,3} = loglik{idx(3),2};
   
   length = size(idx,2);
   midlength = round(length/2);
   BottomPattern{i,1} = loglik{idx(length),2};
   BottomPattern{i,2} = loglik{idx(length-1),2};
   BottomPattern{i,3} = loglik{idx(length-2),2};
   
   MiddlePattern{i,1} = loglik{idx(midlength-1),2};
   MiddlePattern{i,2} = loglik{idx(midlength),2};
   MiddlePattern{i,3} = loglik{idx(midlength+1),2};
   
   
end
