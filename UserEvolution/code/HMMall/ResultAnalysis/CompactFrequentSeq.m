K = 7;
FrequentCompactSeq = [];
WeightFrequentCompactSeq = [];
CompactSeq = zeros(size(author_data_cell,3),15);
WeightCompactSeq = zeros(size(author_data_cell,3),15);

for j = 1:size(author_data_cell,3)
    seq = path{j};
    temp = seq([1,diff(seq)]~=0);
    CompactSeq(j,1:size(temp,2))= temp;
    timestamp = find(diff([-1 seq -1]) ~= 0); % where does V change
    temp = diff(timestamp);
    WeightCompactSeq(j,1:size(temp,2)) = temp;
end

%CompactSeq = cell2mat(CompactSeq');
for i = 1:K
    index = 1;
    Sequences = [];
for j = 1:size(author_data_cell,3)
    if(path_cluster(j) == i)
            Sequences(index,1:size(CompactSeq(j,:),2)) = CompactSeq(j,:);
            WeightSeq(index,1:size(WeightCompactSeq(j,:),2)) = WeightCompactSeq(j,:);
            index = index + 1;
    end
end
for k = 1:size(Sequences,2)
     %Sequences(Sequences==0) = [];
     [M,F,C] = mode(Sequences(:,k));
     if(size(C,1) > 1)
         C
     else
      FrequentCompactSeq(i,k) = M;
      temp = Sequences(:,k);
      %find all the seq which have M value at k
      [ind] = find(temp == M);
      temp = WeightSeq(:,k);
      WeightFrequentCompactSeq(i,k) = mean(temp((ind)));
     end
end
%% Compacting the weighted frequent seq again
seq = FrequentCompactSeq(i,:);
        temp = seq([1,diff(seq)]~=0);
        FrequentCompactSeq(i,:) = 0;
        FrequentCompactSeq(i,1:size(temp,2))= temp;
        timestamp = find(diff([-1 seq -1]) ~= 0);
        temp = diff(timestamp);
        seq = WeightFrequentCompactSeq(i,:);
        WeightFrequentCompactSeq(i,:) = 0;
        for k = 1:size(temp,2)-1
           WeightFrequentCompactSeq(i,k) = sum(seq(timestamp(k):timestamp(k+1)-1));
        end

end


