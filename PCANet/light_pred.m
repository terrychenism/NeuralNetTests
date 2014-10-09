clear all; close all; clc; 

addpath('./Utils');
addpath('./Liblinear');
tic;
ImgX = 28;
ImgY = 14;
NumChls = 3;
filename = 'n.jpg';
load models
V = models.V{1};
TestLabels = 1; % validate
XX = imread(filename);
test_data = double(imresize(XX, [ImgX ImgY]));

%% PCA param 
PatchSize = 7;
NumFilters = 8;
BlkOverLapRatio = 0.5;
HistBlockSize = [7 7];

mag = (PatchSize-1)/2;
OutImg = cell(NumFilters,1); 

img = zeros(ImgX+PatchSize-1,ImgY+PatchSize-1, NumChls);

img((mag+1):end-mag,(mag+1):end-mag,:) = test_data;
im = im2col_general(img,[PatchSize PatchSize]); 
im = bsxfun(@minus, im, mean(im)); 

cnt = 0;
for j = 1:NumFilters
    cnt = cnt + 1;
    OutImg{cnt} = reshape(V(:,j)'*im,ImgX,ImgY);  %% conv
end
ImgIdx = ones(NumFilters,1); 

%% Histogram 
map_weights = 2.^((NumFilters-1):-1:0); 
Idx_span = find(ImgIdx == 1);
 
T = zeros(size(OutImg(Idx_span(1))));      
for j = 1:NumFilters
     T = T + map_weights(j)*Heaviside(OutImg{Idx_span(j)});            
end
       
Bhist = sparse(histc(im2col_general(T,HistBlockSize,...
     round((1- BlkOverLapRatio)*HistBlockSize)),(0:2^NumFilters-1)'));        
Bhist = bsxfun(@times, Bhist, 2^NumFilters./sum(Bhist));         
feature = Bhist(:); % vectorize

%% SVM 

[label_p, accuracy, decision_values] = predict(TestLabels,sparse(feature'), models, '-q'); 
    
%% Results display
PCANet_TestTime = toc;
fprintf('     Predict label: %.1d\n', label_p);
fprintf('     PCANet training time: %.4f secs.\n', PCANet_TestTime);
  
