clear all;
addpath('C:\Users\tchen2\Desktop\stlSubset\liblinear-1.94\matlab');
subjects_train = 1:14;    
disp(strcat('Training on subjects',num2str(subjects_train(1)),':',num2str(subjects_train(end))));

tmin = 0;
tmax = 0.5;
disp(strcat('Restricting MEG data to the interval [',num2str(tmin),num2str(tmax),'] sec.'));
X_train = [];
y_train = [];
X_test = [];
y_test = [];
ids_test = [];

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


%%%%----read test set--------------
% disp('Creating the testset.');
% subjects_test = 4:4;
% for i = 1 : length(subjects_test)
%     path = 'G:/EDU/Coding/Kaggle/DecMeg2014/data/';  % Specify absolute path 
%     filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_test(i));
%     disp(strcat('Loading ',filename));
%     data = load(filename);
%     XX = data.X;
%     yy = data.y;
%     sfreq = data.sfreq;
%     tmin_original = data.tmin;
%     disp('Dataset summary:')
%     disp(sprintf('XX: %d trials, %d channels, %d timepoints',size(XX,1),size(XX,2),size(XX,3)));
%     disp(sprintf('yy: %d trials',size(yy,1)));
%     disp(strcat('sfreq:', num2str(sfreq)));
%     features = createFeatures(XX,tmin, tmax, sfreq,tmin_original);
%     X_test = [X_test;features];
%     y_test = [y_test;yy];
% end

%===================================
disp('Creating the testset.');
subjects_test = 15:16;
for i = 1 : length(subjects_test)
    path = './data/'; % Specify absolute path
    filename = sprintf(strcat(path,'train_subject%02d.mat'),subjects_test(i));
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
    X_test = [X_test;features];
    %ids_test = [ids_test;ids];
    y_test = [y_test;yy];
end
%====================================
%%% step3: Cross Validation for choosing parameter
fprintf(1,'step3: Cross Validation for choosing parameter c...\n');
% the larger c is, more time should be costed
c = [2^-6 2^-5 2^-4 2^-3 2^-2 2^-1 2^0 2^1 2^2 2^3];
max_acc = 0;
tic;
for i = 1 : size(c, 2)
	option = ['-B 1 -c ' num2str(c(i)) ' -v 5 -q'];
	fprintf(1,'Stage: %d/%d: c = %d, ', i, size(c, 2), c(i));
	accuracy = train(double(y_train), sparse(double(X_train)), option);	
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
model = train(double(y_train), sparse(double(X_train)), option);
toc;

fprintf(1,'step5: Testing...\n');
tic;
[predict_label, accuracy, dec_values] = predict(double(y_test),sparse(double(X_test)), model);
toc;


% startTime = cputime;
% model = svmtrain(double(y_train),double(X_train),'-t 0');
% duration = cputime - startTime;
% fprintf('Train Time: %.2f (s)\n', duration);  
% 
% startTime = cputime;
% [predict_label, accuracy, dec_values] = svmpredict(double(y_test),double(X_test), model);
% duration = cputime - startTime;
% fprintf('test Time: %.2f (s)\n', duration);  
% 
% % Saving the results in the submission file:
% filename_submission = 'submission.csv';
% disp(strcat('Creating submission file: ',filename_submission));
% f = fopen(filename_submission, 'w');
% fprintf(f,'%s,%s\n','Id','Prediction');
% for i = 1 : length(y_pred_thresholded)
%     fprintf(f,'%d,%d\n',ids_test(i),y_pred_thresholded(i));
% end
% fclose(f);
% disp('Done.');
