clear all
author_data_cell = {};

temp = [1];
for i = 1:250
    author_data_cell{i} = repmat(temp,[15 1])';
    author_data_cell{i}(1,16:30) = repmat([2],[15 1])';
end

len = size(author_data_cell,2)
temp = [2];
for i = 1:250
    index = len+i;
    author_data_cell{index} = repmat(temp,[15 1])';
    author_data_cell{index}(1,16:30) = repmat([3],[15 1])';
end

len = size(author_data_cell,2)
temp = [3];
for i = 1:250
    author_data_cell{len+i} = repmat(temp,[15 1])';
    author_data_cell{len+i}(1,16:30) = repmat([4],[15 1])';
end

len = size(author_data_cell,2);
temp = [1];
for i = 1:250
    author_data_cell{len+i} = repmat(temp,[15 1])';
    author_data_cell{len+i}(1,16:30) = repmat([4],[15 1])';
end

len = size(author_data_cell,2);
author_data_cell = reshape(author_data_cell, [1 1 len]);