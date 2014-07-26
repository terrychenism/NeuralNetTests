clear; clc;
addpath('./libsvm-3.18/matlab')

%load point.mat
num = 20;

interiorPoints = rand(num,3);  
r=-9:10;
step = size(r);
for j = 1:9
     interiorPoints(j*num+1:j*num+step(2),1) = interiorPoints(1:num,1);
     interiorPoints(j*num+1:j*num+step(2),2) = interiorPoints(1:num,2);
     interiorPoints(j*num+1:j*num+step(2),3) = interiorPoints(1:num,3);
end
interiorPoints = sort(interiorPoints);
for j = 1:9
%% add noise
interiorPoints(1+(j-1)*num:j*num,1) = interiorPoints(1+(j-1)*num:j*num,1) + randn(size(r))';
interiorPoints(1+(j-1)*num:j*num,2) = interiorPoints(1+(j-1)*num:j*num,2) + randn(size(r))';
interiorPoints(1+(j-1)*num:j*num,3) = interiorPoints(1+(j-1)*num:j*num,3) + randn(size(r))';
end

B = sort(interiorPoints);
sum = size(B);
half = sum/2;
for i = 1:half
    B(i,4) = 1;
end

for i = half+1:sum
    B(i,4) = 0;
end



% ===== Reshuffle the training data =====
Randnidx = randperm(size(B,1)); 
B = B(Randnidx,:); 
% =======================================

%% split feature
data = B;
TrnSize = sum*0.6; 
TrnData = data(1:TrnSize, 1:end-1);
TrnLabels = data(1:TrnSize, end);
ValData = data(TrnSize+1:end,1:end-1);
ValLabel = data(TrnSize+1:end, end);

clear data;

%% train
fprintf(1,'Training...\n');
model = svmtrain(double(TrnLabels),double(TrnData),'-t 0');
save('models.mat', 'model');

%% test 
fprintf(1,'Testing...\n');
tic;
[predict_label, accuracy, dec_values] = svmpredict(double(ValLabel),double(ValData), model);
toc;

%% Plot the scattered points:
scatter3(interiorPoints(1:half,1),interiorPoints(1:half,2),interiorPoints(1:half,3),'.','r');
hold
scatter3(interiorPoints(half+1:end,1),interiorPoints(half+1:end,2),interiorPoints(half+1:end,3),'.','b');
axis equal;
title('Interior points');
