function demo

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
caffe.reset_all()
load('voc_gt_cmap.mat');
use_gpu = 1;
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 0;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end

% change the model_def_file and model_file to run your own model
net_model = '../../models/parsenet/parsenet_deploy.prototxt';
net_weights = '../../models/parsenet/VGG_VOC2012ext.caffemodel';
phase = 'test';

net = caffe.Net(net_model, net_weights, phase);


img = imread('cat.jpg');
% seg = wl_segmentImage(img);
base_img(:,:,:,1) = permute(img, [2 1 3]);
octaves = {preprocess(base_img)};
[H, W, C] = size(base_img);
blob_index = net.name2blob_index('data');
net.blob_vec(blob_index).reshape([ H  W 3 1])
net.blobs('data').set_data(octaves{1}); 
net.forward_prefilled();
scores = net.blobs('score').get_data();
% scores = permute(scores, [2 1 3]);
[~, seg] = max(scores, [], 3);
result_seg = uint8(seg - 1);

% imshow(uint8(seg));
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
result_seg_im = reshape(cmap(int32(result_seg)+1,:),[size(result_seg,1),size(result_seg,2),3]);
result_seg_im = permute(result_seg_im, [2 1 3]);
imshow(result_seg_im);    

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

