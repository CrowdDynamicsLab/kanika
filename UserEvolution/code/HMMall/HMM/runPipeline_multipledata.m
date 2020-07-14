clear
clc
tic
addpath(genpath('/home/kanika/HMMall/'))
load s
dataroot = '/Users/kanika/Dropbox/UserEvolution/individual_evolution/MatlabStackExchangeData';
files = dir(dataroot);
directoryNames = {files([files.isdir]).name};
directoryNames = directoryNames(~ismember(directoryNames,{'.','..'}));

%parpool('local',2)

%Run for all datasets
metric = zeros(size(directoryNames,2),1);
for n = 1:size(directoryNames,2)
    dataset = fullfile(dataroot,directoryNames{1,n});
    fprintf('Computing for dataset%s\n',dataset);
    if strcmp(directoryNames{1,n},'money') == 0
        continue
    end
    
    author_data = dlmread(strcat(dataset, '/Stackdata_session.txt'));
    author_index = dlmread(strcat(dataset, '/Stackdata_session.csv'));
    ConvertData
    %sizes = cellfun('length',author_data_cell);
    %mean(sizes), max(sizes), size(sizes)
    EventPrediction
    metric(n) = jsdiv;
    %Perplexity
    %metric(n) = PerplexK;
end
metric, directoryNames
fprintf('Metric value mean and std.dev are %f, %f\n',mean(metric), var(metric));
cleaner = onCleanup(@() delete(gcp('nocreate')));
toc
save '../StackExchangeData/Perplexity_AllStacks_FullGHMM.mat'