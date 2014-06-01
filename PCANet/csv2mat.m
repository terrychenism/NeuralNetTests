csvread('test.csv',1);
mnistTestData = ans;
save('testDataK.mat','mnistTestData');


%%  
>> csvread('train.csv',1);
>> trainData = ans(:,2:end);
>> save('trainDataK.mat','trainData');
>> clear
>> load('trainDataK.mat');
>> csvread('train.csv',1);
>> trainLabels = ans(:,1);
>> save('trainLabelsK.mat','trainLabels');
>> clear
>> load('trainLabelsK.mat');
