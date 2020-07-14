function [dist] = distance(mu0 , i, j)

    Q = size(mu0,2);
    num = (Q*(Q-1))/2;
    dist = zeros(1,num);
    k = 1;
    for i = 1:Q-1
        for j = i+1:Q
            dist(1,k) = (norm((mu0(:,i)- mu0(:,j)),2))^2;
            k = k+1;
        end
    end
end