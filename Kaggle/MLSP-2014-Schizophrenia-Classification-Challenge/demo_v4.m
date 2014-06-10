% This is an example of how to load and access the functional network
% connectivity (FNC) and source-based morphometry (SBM) features, provided
% for the 2014 MLSP Competition, using MATLAB R2011a.
%
% It also includes an example of how to compute the five number summary of
% each feature, as well as group-specific means and standard deviations.
%

%% Load training labels
clear all; 
addpath('./libsvm-3.18/matlab');
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

% Generate five number summary
summary(FNC_train(:,2:end));
% Group means and standard deviations
grpstats(cat(2,FNC_train(:,2:end),dataset({labels_train.Class,'Class'})),'Class',{'mean', 'std'});

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


% num = 40;
% X_train = train(1:num,2:end);
% y_train = labels_train(1:num,2);
% 
% X_test = train(num+1:end,2:end);
% y_test = labels_train(num+1:end,2); 



X_train = train(:,2:end);
y_train = labels_train(:,2);


X_test = test(:,2:end);
%y_test = labels_train(num+1:end,2); 
testsize = size(X_test);
testlabel = testsize(1);
y_test = zeros(testlabel,1);

%% step3: Cross Validation for choosing parameter

fprintf(1,'step3: Cross Validation for choosing parameter c...\n');
% the larger c is, more time should be costed
% c = [2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3];
% max_acc = 0;
% tic;
% for i = 1 : size(c, 2)
% 	option = ['-b 1 -c ' num2str(c(i)) ' -v 5 -q -s 0'];
% 	fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));
% 	accuracy = svmtrain(double(y_train),double(X_train), option);	
% 	if accuracy > max_acc
% 		max_acc = accuracy;
% 		best_c = i;
% 	end
% end
% fprintf(1,'The best c is c = %d.\n', c(best_c));
% toc;



%% step4: train the model
fprintf(1,'step4: Training...\n');
tic;
%option = ['-c ' num2str(c(best_c)) ' -B 1 -e 0.001'];
% option = ['-c ' num2str(c(best_c)) ' -b 1 -e 0.01 -s 0'];
option = ['-c 2 -b 1 -e 0.01 -s 0'];


ids_test = test(:,1); 

% pause;
% C = 1;
%model = svmtrain(double(y_train),double(X_train),'-t 0 -p 0.2 -b 1');

model = svmtrain(double(y_train),double(X_train), option);
toc;


[predict_label, accuracy, prob_estimates] = svmpredict(double(y_test),double(X_test), model,'-b 1');



%pause;

% % Generate five number summary
% summary(SBM_train(:,2:end))
% % % Group means and standard deviations
% grpstats(cat(2,SBM_train(:,2:end),dataset({labels_train.Class,'Class'})),'Class',{'mean', 'std'})
% 
% % Assumes the file 'test_SBM.csv' is in the current folder.
% % Load test SBM features from file into a dataset array variable
% SBM_test = dataset('file','test_SBM.csv','Delimiter',',');

%% Generate new submission file
% Saving the results in the submission file:
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Probability');
for i = 1 : length(prob_estimates)
    fprintf(f,'%d,%.4f\n',ids_test.Id(i),prob_estimates(i));
end
fclose(f);
disp('Done.');
% Assumes the file 'submission_example.csv' is in the current folder.
% Load example submission from file into a dataset array variable
% example = dataset('file','submission_example.csv','Delimiter',',');
% 
% % Compute your scores here: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % SCORES MUST BE VALUES BETWEEN 0 AND 1. Do not submit labels!
% scores = ones(length(predict_label),1);
% 
% % Enter your scored into the example dataset
% example.Probability = prob_estimates;
% 
% % Save your scores in a new submission file.
% % This assumes you have write permission to the current folder.
% export(example,'file','new_submission.csv','Delimiter',',');
