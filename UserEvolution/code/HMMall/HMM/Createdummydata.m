author_data_cell = {};

temp = [0.9 0.1 0 0];
for i = 1:250
    author_data_cell{i} = repmat(temp,[15 1])';
end

len = size(author_data_cell,2)
temp = [0.1 0.9 0 0];
for i = 1:250
    index = len+i;
    author_data_cell{index} = repmat(temp,[15 1])';
end

len = size(author_data_cell,2)
temp = [0 0.1 0.9 0];
for i = 1:250
    author_data_cell{len+i} = repmat(temp,[15 1])';
end

len = size(author_data_cell,2);
temp = [0 0 0.1 0.9];
for i = 1:250
    author_data_cell{len+i} = repmat(temp,[15 1])';
end

len = size(author_data_cell,2);
author_data_cell = reshape(author_data_cell, [1 1 len]);