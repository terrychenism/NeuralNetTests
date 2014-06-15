% This is an example of how to load and access the functional network
% connectivity (FNC) and source-based morphometry (SBM) features, provided
% for the 2014 MLSP Competition, using MATLAB R2011a.
%
% It also includes an example of how to compute the five number summary of
% each feature, as well as group-specific means and standard deviations.
%

%% Load training labels
clear all; 
addpath('D:/DeepLearnToolbox');
% Assumes the file 'train_labels.csv' is in the current folder.
% Load training labels from file into a dataset array variable
labels_train = dataset('file','train_labels.csv','Delimiter',',');

% Convert 'Class' into an unordered categorical variable, and assign labels
% to each level.
% labels_train.Class = nominal(labels_train.Class, {'Healthy Control','Schizophrenic Patient'}, [0, 1]);
% summary(labels_train.Class)

%% Load FNC features
% These are correlation values.

% Assumes the file 'train_FNC.csv' is in the current folder.
% Load training FNC features from file into a dataset array variable
FNC_train = dataset('file','train_FNC.csv','Delimiter',',');

% % Generate five number summary
% summary(FNC_train(:,2:end))
% % Group means and standard deviations
% grpstats(cat(2,FNC_train(:,2:end),dataset({labels_train.Class,'Class'})),'Class',{'mean', 'std'})

% Assumes the file 'test_FNC.csv' is in the current folder.
% Load test FNC features from file into a dataset array variable
FNC_test = dataset('file','test_FNC.csv','Delimiter',',');
% 
% %% Load SBM features
% % These are ICA weights.
% 
% % Assumes the file 'train_SBM.csv' is in the current folder.
% % Load training SBM features from file into a dataset array variable
SBM_train = dataset('file','train_SBM.csv','Delimiter',',');
SBM_test = dataset('file','test_SBM.csv','Delimiter',',');
disp('concatenating all features SBM and FNC.');
% horzcat(SBM_train, FNC_train);
train = horzcat(SBM_train, FNC_train);
test =  horzcat(SBM_test, FNC_test);

clear SMB_train FNC_train SMB_test FNC_test

% 
% num = 70;
% X_train = train(1:num,2:end);
% 
% y_train = labels_train(1:num,2);
% addoneline = zeros(num,1);
% y_train = double(y_train);
% y_train = horzcat(addoneline,y_train);
% 
% 
% X_test = train(num+1:end,2:end);
% y_test = labels_train(num+1:end,2); 
% addoneline = zeros(86-num,1);
% y_test = double(y_test);
% y_test= horzcat(addoneline,y_test);
% 
% X_train = double(X_train);
% X_test = double(X_test);


%% create train set
X_train = train(:,2:end);
y_train = double(labels_train(:,2));
labelSize = size(y_train);
labelNum = labelSize(1);
addoneline = zeros(labelNum,1);
y_train = horzcat(addoneline,y_train);

%% create test set
X_test = test(:,2:end);
% testsize = size(X_test);
% testlabel = testsize(1);
% y_test = zeros(testlabel,1);


%% change to double 
X_train = double(X_train);
X_test = double(X_test);


%% training nn
[X_train, mu, sigma] = zscore(X_train);
X_test = normalize(X_test, mu, sigma);

%% vanilla neural net
rand('state',0)
nn = nnsetup([410 100 2]);
nn.dropoutFraction = 0.5;   %  Dropout fraction 

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs = 300;        %  Number of full sweeps through data
opts.batchsize = 2;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, X_train, y_train, opts);

% [er, bad] = nntest(nn, X_test, y_test);
% assert(er < 0.1, 'Too big error');

labels = nnpredict(nn, X_test) - 1;
ids_test = test(:,1); 

%% Generate new submission file
% Saving the results in the submission file:
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Probability');
for i = 1 : length(labels)
    fprintf(f,'%d,%d\n',ids_test.Id(i),labels(i));
end
fclose(f);
disp('Done.');
