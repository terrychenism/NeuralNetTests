

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
net_model = [model_dir 'very_small_vgg_19_layers.prototxt'];
net_weights = [model_dir 'very_small_vgg_19_layers.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
fileID = fopen('exp.txt','w');
layers = net.layer_names;
for n = 1:size(layers, 1)
    if strcmp(layers{n}(1:2),'fc') || strcmp(layers{n}(1:4),'conv')
        fprintf(fileID, '%s ',layers{n});
        fprintf(fileID, ': ');
        data=net.params(layers{n}, 1).get_data();

        fprintf(fileID, '%d ',size(data));
        fprintf(fileID, '\n');
        for num = 1:size(data,4)
            b = [];
            for w = 1:size(data,1)
                for h = 1:size(data, 2)
                    for c = 1:size(data,3)
        
                        fprintf(fileID, '%f ',data(w,h,c,num));
                    end
                end
                
%                 p = data(:,:,c,num)';
%                 b = p(:)';
%                 fprintf(fileID, '%f ',b);
%                 fprintf(fileID, '| ');
            end
            fprintf(fileID, '\n');
            %cc = [cc; b];

        end
    end
    fprintf(fileID, '\n');
end

% for i = 1:8
%     dlmwrite('a.txt', conv1(:,:,:,i));
% end
