function saliency_map_center

base = [pwd, '/'];
addpath(genpath(base));

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

model_root = 'E:\caffe\caffe-prelu\models\bvlc_reference_caffenet/';
net_model = [model_root , 'deploy.prototxt'];
net_weights = [model_root, 'bvlc_reference_caffenet.caffemodel'];
end_layer = 'fc8';
net = caffe.Net(net_model,net_weights,'test');
im = imread('images/cat.jpg');
[H,W,C] = size(im);

I = im;
[img_height, img_width, ~] = size(I);
pad_offset_col = img_height;
pad_offset_row = img_width;

% pad every images(I, cls_seg, inst_seg...) to make cropping easy
padded_I = padarray(I,[pad_offset_row, pad_offset_col]);
result_base = uint8(zeros(size(I,1), size(I,2)));
padded_result_base = padarray(result_base,[pad_offset_row, pad_offset_col]);

use_edgebox = 1;   
if use_edgebox
    model=load('edgebox/models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 200;  % max number of boxes to detect

    bbs = edgeBoxes(padded_I,model,opts); 
    boxes_padded = xywh_to_bbox(bbs);
    boxes_padded = boxes_padded(:,1:4);
% else
%     cache = load(sprintf(edgebox_cache_path, ids{i})); % boxes_padded
%     boxes_padded = cache.boxes_padded;
end


numBoxes = size(boxes_padded,1);    
result = zeros(H,W);
for bidx = 1:numBoxes 
    box = boxes_padded(bidx,:);
    box_wd = box(3)-box(1)+1;
    box_ht = box(4)-box(2)+1;

    if min(box_wd, box_ht) < 112, continue; end   

    input_data = preprocess_image_bb(padded_I, boxes_padded(bidx,:), 227); 
    
    net.blobs('data').set_data(input_data{1});
    net.forward_prefilled();
    score = net.blobs('prob').get_data();
    [~, maxlabel] = max(score);

    %dst = net.blobs(end_layer).get_data();
    caffeLabel = single(zeros(1000,1));
    caffeLabel(max(maxlabel),:) = 1;

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
    mask = imresize(mask, [box_ht, box_wd], 'bilinear');
    imagesc(mask);
    waitforbuttonpress;
    
    padded_result_base(box(2):box(4),box(1):box(3)) = mean(double(padded_result_base(box(2):box(4),box(1):box(3))),double(mask));
    %result = max(result, mask);
    
end
result = padded_result_base(pad_offset_row:pad_offset_row+size(I,1)-1,pad_offset_col:pad_offset_col+size(I,2)-1,:); 
imagesc(result);
pause;
 



octaves = {prepare_image(im)};
    
        
% octaves = {preprocess(im)};
net.blobs('data').set_data(octaves{1});
net.forward_prefilled();
score = net.blobs('prob').get_data();
[~, maxlabel] = max(score);

%dst = net.blobs(end_layer).get_data();
caffeLabel = single(zeros(1000,10));
caffeLabel(max(maxlabel),:) = 1;

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


% padded_mask = padarray(mask,[pad_offset_row, pad_offset_col]);




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



function preprocessed_img = preprocess_image_bb(img, box, img_sz)
meanImg = [104.00698793, 116.66876762, 122.67891434]; % order = bgr
meanImg = repmat(meanImg, [img_sz^2,1]);
meanImg = reshape(meanImg, [img_sz, img_sz, 3]); 

crop = double(img(box(2):box(4),box(1):box(3),:));
crop = imresize(crop, [img_sz img_sz], 'bilinear'); % resize cropped image
crop = crop(:,:,[3 2 1]) - meanImg; % convert color channer rgb->bgr and subtract mean 
preprocessed_img = {single(permute(crop, [2 1 3]))}; 

end
