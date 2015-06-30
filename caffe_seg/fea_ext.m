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
config.Path.CNN.script_path ='../model/DecoupledNet_Full_anno';
config.Path.CNN.model_data = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference.caffemodel'];
config.Path.CNN.model_proto = [config.Path.CNN.script_path '/DecoupledNet_Full_anno_inference_deploy.prototxt'];


fprintf('start DecoupledNet inference [%s]\n', config.model_name);

%% initialization
load(config.cmap);
% init_VOC2012_TEST;

% initialize caffe
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
img_path = 'F:\VOC2012\VOCdevkit\VOC2012\JPEGImages\';
[names, labels] = textread('F:\VOC2012\VOCdevkit\VOC2012\ImageSets\Main\aeroplane_val.txt', '%s %d');
for i = 1:size(labels,1)
    if labels(i) == 1
    filename = strcat(img_path, names{i,1}, '.jpg');
    I = imread(filename);
    % I=imread('F:\Coding\DecoupledNet\inference\data\VOC2012_TEST\JPEGImages\2008_002695.jpg');

    im_sz = max(size(I,1),size(I,2));
    caffe_im = padarray(I,[im_sz - size(I,1), im_sz - size(I,2)],'post');
    caffe_im = preprocess_image(caffe_im, config.im_sz);
    label = single(zeros([1,1,20]));
    cnn_output = caffe('forward', {caffe_im;label});
    cls_score = cnn_output{1};
    label = single(cls_score .* (cls_score > 0.5));  
    cnn_output = caffe('forward', {caffe_im;label});
    pool5 = cnn_output{size(cnn_output,1)-1, 1};
    [seg, segmask] = max(pool5, [], 3);
    imagesc(seg)
    pause;
    end
end
% image = uint8(zeros(100,100));
% idx = 720;
% for i = 1:10:100
%     for j = 1:10:100
%         image(i:i+9, j:j+9) =   uint8(pool5(:,:,idx));
%         idx = idx+1;
%     end
% end
% imagesc(image);
