loglik = 0;

for i = 1:size(testdata,2)
        data = testdata{i};
        N = size(testdata{i},2);
        Q = path{1,i}(end);
        K = path_cluster(i);
        loglik = loglik + sum(gaussian_prob(data, mu0{K}(:,Q), Sigma0{K}(:,:,Q), 1));

    end
    loglik = loglik/size(testdata,2);
'Loglikelihood is',loglik

