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
im = imread('cat.jpg');
octaves = {preprocess(im)};
net.blobs('data').set_data(octaves{1});
net.forward_prefilled();
dst = net.blobs(end_layer).get_data();

caffeLabel = single(zeros(1000,1));
caffeLabel(281) = 1;

net.blobs(end_layer).set_diff(caffeLabel);
%net.backward_prefilled();
net.backward_from(end_layer);
diff = net.blobs('data').get_diff();
diff = bsxfun(@minus,diff,min(min(diff)));
diff = bsxfun(@rdivide,diff, max(max(diff)));
diff_sq = squeeze(diff);
[mask,~] = max(diff_sq, [], 3);

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
