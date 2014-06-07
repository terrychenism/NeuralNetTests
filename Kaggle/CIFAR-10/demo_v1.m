clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');
CIFAR_DIR='./cifar-10-batches-mat/';
SKIPTO = 2; 
%% Configuration

TrnSize = 20000; 
ImgSize = 32; 
ImgFormat = 'color'; 


%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

trainData = double([f1.data; f2.data; f3.data; f4.data; f5.data]);
trainLabels = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!

clear f1 f2 f3 f4 f5;
TrnData = trainData(1:TrnSize, 1:end)';
TrnLabels = trainLabels(1:TrnSize,end);
ValData = trainData(TrnSize+1:TrnSize+2000,1:end)';
ValLabel = trainLabels(TrnSize+1:TrnSize+2000, end);


% pause;

%% test using subset of trainset 

% TestData = trainData(TrnSize+1:end,1:end)';
% TestLabels = trainLabels(TrnSize+1:end, end);

%% for test dataset
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);
testData = double(f1.data);
testLabels = double(f1.labels) + 1;
clear f1;

TestData = testData(:,1:end)';
TestLabels = testLabels(:,end);
% pause;
% testsize = size(TestData);
% testlabel = testsize(2);
% TestLabels = zeros(testlabel,1);



%% ==== Subsampling the Training and Testing sets ============
% (that for small test, haha! ) 
TrnData = TrnData(:,1:4:end);  % sample around 2500 training samples
TrnLabels = TrnLabels(1:4:end); % 

TestData = TestData(:,1:50:end);  % sample around 1000 test samples  
TestLabels = TestLabels(1:50:end); 

%% ===========================================================

nTestImg = length(TestLabels);

PCANet.NumStages = 2;
PCANet.PatchSize = 7;
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [7 7]; 
PCANet.BlkOverLapRatio = 0.5;

fprintf('\n ====== PCANet Parameters ======= \n')
PCANet

fprintf('\n ====== PCANet Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat); % convert columns in TrnData to cells 

% pause;

clear TrnData; 
tic;
[ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1); % BlkIdx serves the purpose of learning block-wise DR projection matrix; e.g., WPCA
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 

% if SKIPTO <= 1
fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); % we use linear SVM classifier (C = 1), calling libsvm library
LinearSVM_TrnTime = toc;
clear ftrain; 

save('models.mat', 'models');
% else
%   load('models.mat');
% end
%% PCANet Feature Extraction and Testing 

TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); % convert columns in TestData to cells 
clear TestData; 

fprintf('\n ====== PCANet Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 

predict_label = zeros(nTestImg,1);

for idx = 1:1:nTestImg
    
    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet); % extract a test feature using trained PCANet model 

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); % label predictoin by libsvm
    
    predict_label(idx) = xLabel_est;
    
    if xLabel_est == TestLabels(idx)
        RecHistory(idx) = 1;
        nCorrRecog = nCorrRecog + 1;
    end
    
    if 0==mod(idx,nTestImg/100); 
        fprintf('Accuracy up to %d tests is %.2f%%; taking %.2f secs per testing sample on average. \n',...
            [idx 100*nCorrRecog/idx toc/idx]); 
    end 
    
    TestData_ImgCell{idx} = [];
    
end
Averaged_TimeperTest = toc/nTestImg;
Accuracy = nCorrRecog/nTestImg; 
ErRate = 1 - Accuracy;

%% Results display
fprintf('\n ===== Results of PCANet, followed by a linear SVM classifier =====');
fprintf('\n     PCANet training time: %.2f secs.', PCANet_TrnTime);
fprintf('\n     Linear SVM training time: %.2f secs.', LinearSVM_TrnTime);
fprintf('\n     Testing error rate: %.2f%%', 100*ErRate);
fprintf('\n     Average testing time %.2f secs per test sample. \n\n',Averaged_TimeperTest);


%% Saving the results in the submission file
% filename_submission = 'submission.csv';
% disp(strcat('Creating submission file: ',filename_submission));
% f = fopen(filename_submission, 'w');
% fprintf(f,'%s,%s\n','ImageId','Label');
% for i = 1 : length(predict_label)
%     fprintf(f,'%d,%d\n',i,predict_label(i));
% end
% fclose(f);
% disp('Done.');

    
