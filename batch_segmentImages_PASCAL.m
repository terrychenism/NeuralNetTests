function demo

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

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


listing = dir(strcat('C:\Users\cht2pal\Downloads\VOCdevkit\VOC2007\JPEGImages\','*.jpg'));
for i=1:length(listing)
    fprintf('progress: %d/%d ...', i, length(listing));  
    end_layer = 'score';
    img = imread(strcat('C:\Users\cht2pal\Downloads\VOCdevkit\VOC2007\JPEGImages\',listing(i).name));
    base_img = permute(img, [2 1 3]);
    octaves = {preprocess(base_img)};
    [H, W, C] = size(base_img);
    blob_index = net.name2blob_index('data');
    net.blob_vec(blob_index).reshape([ H  W 3 1])
    net.blobs('data').set_data(octaves{1}); 
    % net.forward_prefilled();
    tic;
    net.forward_to(end_layer);
    toc;
    % scores = net.blobs(end_layer).get_data();

    dst = net.blobs(end_layer).get_data();
    net.blobs(end_layer).set_diff(dst);
    % net.backward_prefilled();
    net.backward_fromto('score','pool5');
    g = net.blobs('pool5').get_diff();

    % net.backward_from(end_layer);
    % diff = net.blobs('data').get_diff();
    % diff = bsxfun(@minus,diff,min(min(diff)));
    % diff = bsxfun(@rdivide,diff, max(max(diff)));
    % diff_sq = squeeze(diff);
    % [mask,~] = max(diff_sq, [], 4);
    % im_data = permute(mask, [2, 1, 3]);
    % [mask,~] = max(im_data, [], 3);
    % mask = imresize(mask,[H W]);
    % imagesc(mask);

    [~, seg] = max(dst, [], 3);
    result_seg = uint8(seg - 1);

    % imshow(uint8(seg));
    show = 0;
    result_seg_im = reshape(cmap(int32(result_seg)+1,:),[size(result_seg,1),size(result_seg,2),3]);
    result_seg_im = permute(result_seg_im, [2 1 3]);
    if show == 1
        subplot(1,2,1);
        imshow(img);
        subplot(1,2,2);
        imshow(result_seg_im);   
    end
    save_res_path = 'result/';
    imwrite(result_seg_im, [save_res_path, listing(i).name]);
end

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
