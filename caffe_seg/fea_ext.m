clear all; close all; clc;

%% startup
startup;
config.imageset = 'test';
config.cmap= './voc_gt_cmap.mat';       
config.gpuNum = 0;                      % gpu id
config.Path.CNN.caffe_root = './caffe'; % caffe root path
config.save_root = './results';         % result will be save in this directory

%% configuration
config.write_file = 0;
config.thres = 0.5;
config.im_sz = 320;

%% DecoupledNet Full annotations
config.model_name = 'DecoupledNet_Full_anno';
config.Path.CNN.script_path = 'C:/Users/cht2pal/Desktop/caffe-old-unpool/examples/DecoupledNet';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference_deploy.prototxt'];


fprintf('start DecoupledNet inference [%s]\n', config.model_name);

%% initialization
load(config.cmap);
% init_VOC2012_TEST;

% initialize caffe
% addpath(fullfile(config.Path.CNN.caffe_root, 'matlab/caffe'));
fprintf('initializing caffe..\n');
if caffe('is_initialized')
    caffe('reset')
end
caffe('init', config.Path.CNN.model_proto, config.Path.CNN.model_data)
caffe('set_device', config.gpuNum);
caffe('set_mode_gpu');
caffe('set_phase_test');
fprintf('done\n');

%% initialize paths
save_res_dir = sprintf('%s/%s',config.save_root ,config.model_name);
save_res_path = [save_res_dir '/%s.png'];

%% create directory
if config.write_file
    if ~exist(save_res_dir), mkdir(save_res_dir), end
end

fprintf('start generating result\n');
fprintf('caffe model: %s\n', config.Path.CNN.model_proto);
fprintf('caffe weight: %s\n', config.Path.CNN.model_data);

I=imread('000079.jpg');
  
im_sz = max(size(I,1),size(I,2));
caffe_im = padarray(I,[im_sz - size(I,1), im_sz - size(I,2)],'post');
caffe_im = preprocess_image(caffe_im, config.im_sz);
label = single(zeros([1,1,20]));
label(1,1,8) = 1;
cnn_output = caffe('forward', {caffe_im;label});

pool5 = cnn_output{size(cnn_output,1)-1, 1};
image = uint8(zeros(100,100));
idx = 1;
for i = 1:10:100
    for j = 1:10:100
        image(i:i+9, j:j+9) =   uint8(pool5(:,:,idx));
        idx = idx+1;
    end
end
imagesc(image);
