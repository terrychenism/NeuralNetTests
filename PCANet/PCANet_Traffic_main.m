clear all; close all; clc; 
addpath('./Utils');
addpath('./Liblinear');


TrnSize = 2000; 
ImgSize = 28; 
ImgFormat = 'gray'; 


% load data
load data_traffic
mnist_train = data;
clear train;
% ===== Reshuffle the training data =====
Randnidx = randperm(size(mnist_train,1)); 
mnist_train = mnist_train(Randnidx,:); 
% =======================================

TrnData = mnist_train(1:TrnSize,1:end-1)';  % partition the data into training set and validation set
TrnLabels = mnist_train(1:TrnSize,end);
TestData = mnist_train(TrnSize+1:end,1:end-1)';
TestLabels = mnist_train(TrnSize+1:end,end);
clear mnist_train;

load test
mnist_test = test;
clear test;

TestData = mnist_test(:,1:end-1)';
TestLabels = mnist_test(:,end);
clear mnist_test;


% ==== Subsampling the Training and Testing sets ============
TrnData = TrnData(:,1:4:end);  
TrnLabels = TrnLabels(1:4:end);  
TestData = TestData(:,1:50:end);  
TestLabels = TestLabels(1:50:end); 
% ===========================================================

nTestImg = length(TestLabels);


PCANet.NumStages = 2;
PCANet.PatchSize = 7;
PCANet.NumFilters = [8 8];
PCANet.HistBlockSize = [7 7]; 
PCANet.BlkOverLapRatio = 0.5;

fprintf('\n ====== PCANet Parameters ======= \n')
PCANet


fprintf('\n ====== PCANet Training ======= \n')
TrnData_ImgCell = mat2imgcell(TrnData,ImgSize,ImgSize,ImgFormat);
clear TrnData; 
tic;
[ftrain V BlkIdx] = PCANet_train(TrnData_ImgCell,PCANet,1); 
PCANet_TrnTime = toc;
clear TrnData_ImgCell; 


fprintf('\n ====== Training Linear SVM Classifier ======= \n')
tic;
models = train(TrnLabels, ftrain', '-s 1 -q'); 
LinearSVM_TrnTime = toc;
clear ftrain; 



TestData_ImgCell = mat2imgcell(TestData,ImgSize,ImgSize,ImgFormat); 
clear TestData; 

fprintf('\n ====== Testing ======= \n')

nCorrRecog = 0;
RecHistory = zeros(nTestImg,1);

tic; 

predict_label = zeros(nTestImg,1);
for idx = 1:1:nTestImg
    
    ftest = PCANet_FeaExt(TestData_ImgCell(idx),V,PCANet); 

    [xLabel_est, accuracy, decision_values] = predict(TestLabels(idx),...
        sparse(ftest'), models, '-q'); 
   
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



    
