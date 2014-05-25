clear all;
addpath('D:/libsvm-3.18/matlab')
subjects_train = 1:3;    
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
    path = 'G:/EDU/Coding/Kaggle/DecMeg2014/data/';  % Specify absolute path 
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
subjects_test = 17:18;
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
%====================================

startTime = cputime;
model = svmtrain(double(y_train),double(X_train),'-t 0');
duration = cputime - startTime;
fprintf('Train Time: %.2f (s)\n', duration);  

startTime = cputime;
[predict_label, accuracy, dec_values] = svmpredict(double(y_test),double(X_test), model);
duration = cputime - startTime;
fprintf('test Time: %.2f (s)\n', duration);  

% Saving the results in the submission file:
filename_submission = 'submission.csv';
disp(strcat('Creating submission file: ',filename_submission));
f = fopen(filename_submission, 'w');
fprintf(f,'%s,%s\n','Id','Prediction');
for i = 1 : length(y_pred_thresholded)
    fprintf(f,'%d,%d\n',ids_test(i),y_pred_thresholded(i));
end
fclose(f);
disp('Done.');
