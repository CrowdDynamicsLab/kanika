% path = state decoding variable
K = 4;
%% Creating a compact length path 
compactpath={};
weightpath={};
samelencompactpath={};
for i =1:size(path,2)
    seq = path{i};
    temp = seq([1,diff(seq)]~=0);
    timestamp = find(diff([-1 seq -1]) ~= 0); % where does V change
    dif = diff(timestamp);
    temp=temp(dif>1); % retain only those symbols which are more than 1
    dif=dif(dif>1);
    %compactpath(i,1:size(temp,2))= temp;
    %weightpath(i,1:size(dif,2))=dif;
    compactpath{i}=temp;
    weightpath{i}=dif;
    
end

%%
if K > 1
    for i = 1:K
    index = 1;
    Sequences = [];
    for j = 1:size(author_data_cell,2)
        if(path_cluster(j) == i)
            Sequences(index,1:size(compactpath{j},2)) = compactpath{j};
            WeightSeq(index,1:size(weightpath{j},2)) = weightpath{j};
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
end

%% Find length of each cluster
% K = 4;
% 
% for i = 1:K
%     lengtharr = cell2mat(cellfun(@(x)size(x,2),path(find(Cluster_Assign==i)),'UniformOutput',false));
%     histogram(lengtharr);
%     fprintf('%d,%f,%f,%f\n',i,mean(lengtharr),max(lengtharr),min(lengtharr),median(lengtharr));
%     figure;
% end

