clear; clc;
addpath('C:\Users\tchen2\Desktop\stlSubset\liblinear-1.94\matlab')
%%% step1: load data
fprintf(1,'step1: Load data...\n');

load cnnPooledFeatures.mat;
load stlTrainSubset.mat % loads numTrainImages, trainImages, trainLabels
load stlTestSubset.mat  % loads numTestImages,  testImages,  testLabels

% B = permute(A,order) 
train_X = permute(pooledFeaturesTrain, [1 3 4 2]);

train_X = reshape(train_X, numel(pooledFeaturesTrain) / numTrainImages, numTrainImages);
train_Y = trainLabels; % 2000*1

test_X = permute(pooledFeaturesTest, [1 3 4 2]);
test_X = reshape(test_X, numel(pooledFeaturesTest) / numTestImages, numTestImages);
test_Y = testLabels;
% release some memory
clear trainImages testImages pooledFeaturesTrain pooledFeaturesTest;

%%% step2: scale the data
fprintf(1,'step2: Scale data...\n');
% Using the same scaling factors for training and testing sets, 
% we obtain much better accuracy. Note: scale each attribute(feature), not sample
% scale to [0 1]
% when a is a vector, b = (a - min(a)) .* (upper - lower) ./ (max(a)-min(a)) + lower
lower = 0;
upper = 1.0;
train_X = train_X';
X_max = max(train_X);
X_min = min(train_X);
train_X = (train_X - repmat(X_min, size(train_X, 1), 1)) .* (upper - lower) ...
			./ repmat((X_max - X_min), size(train_X, 1), 1) + lower;
test_X = test_X';
test_X = (test_X - repmat(X_min, size(test_X, 1), 1)) .* (upper - lower) ...
			./ repmat((X_max - X_min), size(test_X, 1), 1) + lower;
% Note: before scale the accuracy is 80.4688%, after scale it turns to 80.1875%,
% and took more time. So is that my scale operation wrong or other reasons?
% After adding bias, Accuracy = 80.75% (2584/3200)

%%% step3: Cross Validation for choosing parameter
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');
% the larger c is, more time should be costed
c = [2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3];
max_acc = 0;
tic;
for i = 1 : size(c, 2)
	option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];
	fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));
	accuracy = train(train_Y, sparse(train_X), option);	
	if accuracy > max_acc
		max_acc = accuracy;
		best_c = i;
	end
end
fprintf(1,'The best c is c = %d.\n', c(best_c));
toc;

%%% step4: train the model
fprintf(1,'step4: Training...\n');
tic;
option = ['-c ' num2str(c(best_c)) ' -B 1 -e 0.001'];
model = train(train_Y, sparse(train_X), option);
toc;

%%% step5: test the model
fprintf(1,'step5: Testing...\n');
tic;
[predict_label, accuracy, dec_values] = predict(test_Y, sparse(test_X), model);
toc;
