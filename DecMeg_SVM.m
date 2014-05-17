clear all;

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
% Crating the trainset. (Please specify the absolute path for the train data)
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
X_train = X_train(:,1:5000);
%y_train = y_train(:,:);
%-------------------
fprintf('read data succeess. Press enter to continue.\n');
pause;

%% ===========  PCA   ===================

% fprintf(['\nRunning PCA on face dataset.\n' ...
%          '(this mght take a minute or two ...)\n\n']);
% 
% %  Before running PCA, it is important to first normalize X by subtracting 
% %  the mean value from each feature
% [X_norm, mu, sigma] = featureNormalize(X_train);
% 
% %  Run PCA
% [U, S] = pca(X_norm);
% 
% 
% K = 100;
% Z = projectData(X_norm, U, K);
% 
% fprintf('The projected data Z has a size of: ')
% fprintf('%d ', size(Z));
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;
% 
% %------------------------
% plotData(Z, y_train);
plotData(X_train, y_train);
startTime = cputime;
%% ==================== Training Linear SVM ====================
C = 1;
sigma = 0.1;
x1 = [1 2 1]; x2 = [0 4 -1]; 

model = svmTrain(X_train, single(y_train), C, @linearKernel, 1e-3, 20);


%% =============== Implementing RBF Kernel ===============

% fprintf('\nEvaluating the Gaussian Kernel ...\n')
% x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
% sim = gaussianKernel(x1, x2, sigma);
% % SVM Parameters
% C = 1; sigma = 0.1;
% model= svmTrain(X_train, single(y_train), C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%% ======================== end ====================

%%%%----read test set--------------
disp('Creating the testset.');
subjects_test = 4:4;
for i = 1 : length(subjects_test)
    path = 'G:/EDU/Coding/Kaggle/DecMeg2014/data/';  % Specify absolute path 
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
    y_test = [y_test;yy];
end
X_test = X_test(1:594,1:5000);
y_test = y_test(1:594,:);


%  Run PCA

% [X_norm, mu, sigma] = featureNormalize(X_test);
% [U_test, S] = pca(X_norm);
% K = 100;
% Z_test = projectData(X_norm, U_test, K);
%     
% % plotData(X_train, y_train);
% startTime = cputime;
% C = 0.1;
% sigma = 0.1;
% x1 = [1 2 1]; x2 = [0 4 -1]; 
% model= svmTrain(X_train, single(y_train), C, @(x1, x2) gaussianKernel(x1, x2, sigma));
% %C = 0.1;
% %model = svmTrain(X_train, single(y_train), C, @linearKernel);
duration = cputime - startTime;
fprintf('Train Time: %.2f (s)\n', duration);  
% 
fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
% 
p = svmPredict(model, X_test);
% 
fprintf('Test Accuracy: %f\n', mean(double(p == y_test)) * 100);
% pause;


%visualizeBoundary(X_train, single(y_train), model);
