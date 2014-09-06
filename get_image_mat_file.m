clear all; close all; clc
load labels.mat
skip = 0;
if skip ~= 1
    s = 28;
    srcFiles= getImageSet('data\train') ;
    Ntake = length(srcFiles);
    train = double(zeros(Ntake,s*s+1) );
    name = 'data\train\';
    for i = 1:Ntake
        fprintf('%d/%d...\n', i, Ntake);

        
        filename = [name,int2str(i),'.bmp'];
%         filename= char(srcFiles(i));
        XX = imread(filename);
        img_size = size(XX);
        if  size(img_size,2) == 3
            XX = rgb2gray(XX);
        end
        XX = imresize(XX, [s s]);
        X = double(XX);      
        XX = normr(X);
        X = 1 - XX;
        B = reshape(X,1,[]); 
        train(i,1:end-1) = B(1:s*s);
        train(i,end) = labels(i);   
    end
    save('train.mat','train');
else
    load train.mat
end
% [C1] = csvimport( 'trainLabels.csv', 'Class', {'C2'} );
% labels = [1;1;1;1;1;1;1;1;1;2;2;2;2;2;2;2;2;2];
% labels = uint8(labels);
% data  = [];
% data = [features,labels];
% 
% save('data.mat','data');
