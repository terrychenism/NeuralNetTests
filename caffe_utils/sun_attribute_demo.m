function [scores, maxlabel] = sun_attribute_demo(im, use_gpu)

% Add caffe/matlab to you Matlab search PATH to use matcaffe
if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

use_gpu = 1;
% Set caffe mode
if exist('use_gpu', 'var') || use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
model_dir = 'model/';
net_model = [model_dir 'sun_deploy.prototxt'];
net_weights = [model_dir 'sun_attribute.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
im = imread('sunny.jpg');


tic;
input_data = {preprocess(im)};
toc;

% do forward pass to get scores
% scores are now Channels x Num, where Channels == 1000
tic;
% The net forward function. It takes in a cell array of N-D arrays
% (where N == 4 here) containing data of input blob(s) and outputs a cell
% array containing data from output blob(s)
scores = net.forward(input_data);
toc;

scores = scores{1};
score = scores(:,1) < scores(:,2);
% scores = mean(scores, 2);  % take average scores over 10 crops
% 
% [~, maxlabel] = max(scores);

% call caffe.reset_all() to reset caffe
caffe.reset_all();

end

function crops_data = preprocess(im)
crops_data = zeros(224, 224, 3, 1, 'single');


    if size(im, 3) == 1
        im = repmat(im,[1 1 3]);
    end
    im = imresize(im, [224,224]);
%     [h,w,c] = size(im);
    mean_pix =  [104.0, 116.0, 122.0];
    im_data = im(:, :, [3 2 1]);  % permute channels from RGB to BGR
    im_data = single(im_data);  % convert from uint8 to single
%     crops_data(:,:,:,i) = im_data;
    
    for c = 1:3
        crops_data(:, :, c, 1) = im_data(:,:,c) - mean_pix(c);
    end
    
end

