% function noreturn=fumin_algo(filename)
filename = 'CIMG6911.jpg';
ImgSize = 32; 
ImgFormat = 'color'; 

addpath('./Liblinear');
load('models.mat');
load('v_param.mat');
load('PCANet.mat');
% if isempty(models)
%     load('models.mat');
% end

img = imread(filename);
img = double(img);
img = imresize(img,[32,32]); 


img = reshape(img,1,32*32*3)';
TestData_ImgCell = mat2imgcell(img,ImgSize,ImgSize,ImgFormat);

id = 2;
ftest = PCANet_FeaExt(TestData_ImgCell,V,PCANet); 
[labels, accuracy, decision_values] = predict(id,sparse(ftest'), models, '-q'); 

switch labels
  case 1
    fprintf('airplane \n');
  case 2
    fprintf('automobile \n');
  case 3
    fprintf('bird \n');
  case 4
    fprintf('cat \n');
  case 5
    fprintf('deer \n');
  case 6
    fprintf('dog \n');
  case 7
    fprintf('frog \n');
  case 8
    fprintf('horse \n');
  case 9
    fprintf('ship \n');
  otherwise
    fprintf('truck ¥d¨®\n');
end
% noreturn=1;
