function saliency_map

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end
caffe.reset_all();
caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);
% mean_pix =  [104.0, 116.0, 122.0];

model_root = 'G:/EDU/_SOURCE_CODE/caffe/new_caffe/caffe-windows/models/bvlc_reference_caffenet/';
net_model = 'deploy_fc8.prototxt';
net_weights = [model_root, 'bvlc_reference_caffenet.caffemodel'];
end_layer = 'fc8';
net = caffe.Net(net_model,net_weights,'test');
im = imread('images/cat.jpg');
[H, W, C] = size(im);
% blob_index = net.name2blob_index('data');
% net.blob_vec(blob_index).reshape([ H  W 3 1])

% octaves = {preprocess(im)};
octaves = {prepare_image(im)};
net.blobs('data').set_data(octaves{1});
net.forward_prefilled();
scores = net.blobs('prob').get_data();
scores = mean(scores, 2); 
[~, maxlabel] = max(scores);
% dst = net.blobs(end_layer).get_data();

caffeLabel = single(zeros(1000,10));
caffeLabel(maxlabel,:) = 1;

net.blobs(end_layer).set_diff(caffeLabel);
%net.backward_prefilled();
net.backward_from(end_layer);
diff = net.blobs('data').get_diff();
diff = bsxfun(@minus,diff,min(min(diff)));
diff = bsxfun(@rdivide,diff, max(max(diff)));
diff_sq = squeeze(diff);
[mask,~] = max(diff_sq, [], 4);
im_data = permute(mask, [2, 1, 3]);
[mask,~] = max(im_data, [], 3);
mask = imresize(mask,[H W]);
imagesc(mask);

end



function crops_data = preprocess(im)
    IMAGE_SIZE = 227;
    im = imresize(im, [IMAGE_SIZE IMAGE_SIZE]);
    [h,w,c] = size(im);
    mean_pix =  [104.0, 116.0, 122.0];
    im_data = im(:, :, [3 2 1]);  % permute channels from RGB to BGR
    im_data = single(im_data);  % convert from uint8 to single
    crops_data = zeros(h, w, 3, 1, 'single');
    for c = 1:3
        crops_data(:, :, c, :) = im_data(:,:,c) - mean_pix(c);
    end
end



% ------------------------------------------------------------------------
function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)

% oversample (4 corners, center, and their x-axis flips)
crops_data = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
n = 1;
for i = indices
  for j = indices
    crops_data(:, :, :, n) = im_data(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :);
    crops_data(:, :, :, n+5) = crops_data(end:-1:1, :, :, n);
    n = n + 1;
  end
end
center = floor(indices(2) / 2) + 1;
crops_data(:,:,:,5) = ...
  im_data(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:);
crops_data(:,:,:,10) = crops_data(end:-1:1, :, :, 5);

end
