function fcn_demo

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

load('voc_gt_cmap.mat');

% use_gpu = 1;
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
% model_dir = '../../models/bvlc_reference_caffenet/';
net_model = 'deploy.prototxt';
% net_weights = [model_dir 'bvlc_reference_caffenet.caffemodel'];

net_weights = 'resnet_poly_iter_10000.caffemodel';

phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);


img = imread('G:\caffe_pkg\caffe-seg\images/1.jpg');
img = imresize(img,0.5);
end_layer = 'upscore';
base_img(:,:,:,1) = permute(img, [2 1 3]);
octaves = {preprocess(base_img)};
[H, W, C] = size(base_img);
blob_index = net.name2blob_index('data');
net.blob_vec(blob_index).reshape([ H  W 3 1])
net.blobs('data').set_data(octaves{1}); 

tic;
net.forward_prefilled();
toc;

map = net.blobs(end_layer).get_data();

[~, result_seg] = max(map,[], 3);   
result_seg = uint8(result_seg-1);


subplot(1,2,1);
imshow(img);
subplot(1,2,2);
result_seg_im = reshape(cmap(int32(result_seg)+1,:),[size(result_seg,1),size(result_seg,2),3]);
result_seg_im = permute(result_seg_im, [2 1 3]);
imshow(result_seg_im);    

caffe.reset_all();
end





function crops_data = preprocess(im)
    [h,w,c] = size(im);
    mean_pix =  [104.0, 116.0, 122.0];
    im_data = im(:, :, [3 2 1]);  % permute channels from RGB to BGR
    im_data = single(im_data);  % convert from uint8 to single
    crops_data = zeros(h, w, 3, 1, 'single');
    for c = 1:3
        crops_data(:, :, c, :) = im_data(:,:,c) - mean_pix(c);
    end
end

