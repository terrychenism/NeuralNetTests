%% Load training labels
clear all; close all; clc; 
addpath('C:/Users/tchen2/Desktop/Kaggle/DeepLearnToolbox');

subjects_train = 1:16;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));

tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
X_train = [];
y_train = [];
X_test = [];
y_test = [];
ids_test = [];

%% create train set 
disp('Creating the trainset.');
for i = 1 : length(subjects_train)
    path = './data/'; 
    filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_train(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    yy = data.y;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('yy: %d trials',size(yy,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_train = [X_train;features];
    y_train = [y_train;yy];
    
end
num = 9400; 
X_train = X_train(1:num,:);
y_train = y_train(1:num,:);


labelSize = size(y_train);
labelNum = labelSize(1);
addoneline = zeros(labelNum,1);
y_train = horzcat(addoneline,y_train);
%% -----------------------------

% labelSize = size(y_train);
% labelNum = labelSize(1);
% addoneline = zeros(labelNum,1);
% y_train = horzcat(addoneline,y_train);
% 
% num = 3000; 
% train_x = X_train(1:num,:);
% train_y = y_train(1:num,:);
% 
% test_x = X_train(num+1:end,:);
% test_y = y_train(num+1:end,:);
% clear X_train y_train;

%=====read test set===========
disp('Creating the testset.');
subjects_test = 17:23;
for i = 1 : length(subjects_test)
    path = './data/'; % Specify absolute path
    filename = sprintf(strcat(path,'test_subject%02d.mat'),subjects_test(i));
    disp(strcat('Loading ',filename));
    data = load(filename);
    XX = data.X;
    ids = data.Id;
    sfreq = data.sfreq;
    tmin_original = data.tmin;
    disp('Dataset summary:')
    disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
    disp(sprintf('Ids: %d trials',size(ids,1)));
    disp(strcat('sfreq:', num2str(sfreq)));
    features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
    X_test = [X_test;features];
    ids_test = [ids_test;ids];
end

 
%% change to double 
% train_x = double(train_x);
% test_x = double(test_x);
% train_y = double(train_y);
% test_y = double(test_y);

X_train = double(X_train);
X_test = double(X_test);
y_train = double(y_train);

% pause;
%% training nn
[X_train, mu, sigma] = zscore(X_train);
X_test = normalize(X_test, mu, sigma);


%% vanilla neural net
rand('state',0)
nn = nnsetup([38250 100 2]);
nn.dropoutFraction = 0.5;   %  Dropout fraction 
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs = 10;        %  Number of full sweeps through data
opts.batchsize = 50;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, X_train, y_train, opts);
% load('model.mat');
% [er, bad] = nntest(nn, test_x, test_y);
% assert(er < 0.1, 'Too big error');

labels = nnpredict(nn, X_test) - 1;
% ids_test = test(:,1); 



filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(labels)
    fprintf(f,'%d,%d\n',ids_test(i),labels(i));
end
fclose(f);
disp('Done.');


