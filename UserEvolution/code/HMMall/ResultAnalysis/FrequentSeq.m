%function [K] = FrequentSeq(K) 

K = 7; %Cluster size
FrequentSequence = [];

for i = 1:K
    index = 1;
    Sequences = [];
for j = 1:size(author_data_cell,3)
    if(path_cluster(j) == i)
            Sequences(index,1:size(path{j},2)) = path{j};
            index = index + 1;
    end
end
for k = 1:size(Sequences,2)
     %Sequences(Sequences==0) = [];
     [M,F,C] = mode(Sequences(:,k));
     if(size(C,1) > 1)
         C
     else
      FrequentSequence(i,k) = M;
     end
end
end
%end
