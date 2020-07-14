%clear all
%clc
%convert python data into demo format
%fileId = fopen('/Users/kanika/Dropbox/UserEvolution/individual_evolution/InterFiles/Authors_15year_microsoft.csv');
%author_data = dlmread('/Users/kanika/Dropbox/UserEvolution/individual_evolution/InterFiles/Authordata_15year_microsoft.txt');
%author_index = textscan(fileId,'%s\t%f\t%f\n');
%author_data = dlmread('../../StackExchangeData/MatlabData/Stackdata_session.txt');
%author_index = dlmread('../../StackExchangeData/MatlabData/Stackdata_session.csv');

%author_data = dlmread('../../movieStackExchangeData/MatlabData/Stackdata_session.txt');
%author_index = dlmread('../../movieStackExchangeData/MatlabData/Stackdata_session.csv');

author_data_cell = {};

k = 1;

 for i = 1:size(author_index,1)
     span = author_index(i,3)- author_index(i,2)+1;
     if(span >= 10 && span <= 750)
         author_data_cell{k}=author_data(author_index(i,2):author_index(i,3),:)';
         k = k+1;
     %else
        % i
     end
 end

fprintf('Number of Users %d\n',k);

% for i = 1:length(author_index{1,2})
%    span = author_index{1,3}(i)- author_index{1,2}(i)+1;
%    if(span >= 15)
%        author_data_cell{k}=author_data(author_index{1,2}(i):author_index{1,2}(i)+14,1:5)';
%        k = k+1;
%    else
%        i
%    end
% end


%author_data_cell = reshape(author_data_cell,[1 1 k-1]);
%author_data = cell2mat(author_data_cell);
