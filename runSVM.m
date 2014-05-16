data =  load('./data/data.2.mat');
% data = load('./data/train_subject01.mat');
% X = data.data.test.inputs;
% y = data.data.test.targets;
%% SVM Parameters
Cs = [0.1]; 
for c_index=1:4
    Cs = [Cs, Cs(end) * 4];
end

% sigmas = [500];
% sigmas = [1:10:100];
sigmas = [1, 3, 5, 10, 30, 50, 100];

 %channels = [1:3:306, 2:3:306];
 channels = [1:3:306];         % Only get from the magnetometer

% Init Saving Data
% PerReg_CV = zeros(length(Cs), length(sigmas));
% PerReg = zeros(length(Cs), length(sigmas));
% MeanReg = zeros(length(Cs), length(sigmas));
PerReg_CV = zeros(length(channels), length(Cs), length(sigmas));
PerReg = zeros(length(channels), length(Cs), length(sigmas));
MeanReg = zeros(length(channels), length(Cs), length(sigmas));
%%---------test---
X = data.data.training.inputs';
% X = getChannels(X, channels);
X = featureScaling(X);
y = data.data.training.targets';
y = (vec2ind(y') -1)';

x1 = [1 2 1]; x2 = [0 4 -1];
%%---------end---
% X = data.training.inputs';
% X = getChannels(X, channels);
% X = featureScaling(X);
% y = data.training.targets';
% y = (vec2ind(y') -1)';
% 
X_cv = data.data.validation.inputs';
%X_cv = getChannels(X_cv, channels);
X_cv = featureScaling(X_cv);
y_cv = data.data.validation.targets';
y_cv = (vec2ind(y_cv') -1)';
% C = Cs(c_index);    
% for s_index = 1:length(sigmas)
%     sigma = sigmas(s_index);            
%     fprintf('Channel = %d\tC = %f\tSigma = %f\n', channels, C, sigma);           
%     
           


    for ch_index = 1:length(channels)
        channel = channels(ch_index);


        for c_index = 1:length(Cs)
            C = Cs(c_index);    
            for s_index = 1:length(sigmas)
                sigma = sigmas(s_index);            
                fprintf('Channel = %d\tC = %f\tSigma = %f\n', channel, C, sigma);
                
                startTime = cputime;
                model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
                %model=svmTrain(X, y, C, @linearKernel, 1e-3, 20);
                duration = cputime - startTime;
                fprintf('Train Time: %.2f (s)\n', duration);                    

                % Traing Performance
                y_pred= svmPredict(model, X);
                per = mean(y_pred == y);
                PerReg(c_index, s_index) = per;
                fprintf('Training Performance: %f\n', per);

                % Cross-Validation Performance
                y_cv_pred = svmPredict(model, X_cv);
                per_cv = mean(y_cv_pred == y_cv);
                PerReg_CV(c_index, s_index) = per_cv;

                meanPerCV = mean(y_cv_pred);
                MeanReg(c_index, s_index) = meanPerCV;

                fprintf('Cross-Validation Performance: %f\t(Mean Predict = %f) (Mean Label = %f)\n\n', per_cv, meanPerCV, mean(y_cv));
            end
        end
end

close all;
figure; surf(PerReg); title('Performance on Training Data');
figure; surf(PerReg_CV); title('Performance on Validation Data');
figure; surf(MeanReg); title('Mean of Predict on Validation Data');

