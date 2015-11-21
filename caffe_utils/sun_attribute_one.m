function sun_attribute

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

caffe.set_mode_gpu();
gpu_id = 0;
caffe.set_device(gpu_id);
mean_pix =  [104.0, 116.0, 122.0];

net_model = 'sun_deploy.prototxt';

attributes = [];
for j = 1:102
    
% net_weights = strcat('../../examples/sun_attributes/models/sun_attribute_train_list_', int2str(j),'.txt.caffemodel');
net_weights = strcat('../../examples/sun_attributes/models/','sun_attribute_87.caffemodel');


[img_name, label] = textread('val.txt','%s %s',20050);

net = caffe.Net(net_model,net_weights,'test');
batchsize = 250;
attributes_one = [];
% for i = 1:80
%     fprintf('batch: %d\n',i);
%     input_data = {};
%     for b = 1:batchsize
%         img_full_name(b) =  strcat('/media/tairuichen/Data/PLACE2/ILSVRC2015_img_val/',img_name((i-1)*80+b));
%     end
%     %im = imread(img_full_name{1});
%     input_data = {preprocess(img_full_name, batchsize)};
%     scores = net.forward(input_data);
% 
% 
%     scores = scores{1};
%     %scores = mean(scores, 2);  % take average scores over 10 crops
% 
% %     [~, maxlabel] = max(scores);
% %     if maxlabel == 1
% %         attribute = 0;
% %     else
% %         attribute = 1;
% %     end
% %     if attribute == 1
% %         pause;
% %     end
% 
%     attribute = scores(2,:)';
%     attributes_one = [attributes_one; attribute];
%     
% end

%[img_name, y] = textread('/home/tairuichen/Desktop/SUNAttributeDB/list_files/train_list_76.txt','%s %s');
% folder_name = '/media/tairuichen/Data/images/';
folder_name = '/media/tairuichen/Data/weather_database/cloudy/';
img_full_name = [];
img_name = dir(strcat(folder_name,'*.jpg'));
input_data = {};
batch_size = 10;
for b = 1:batch_size
    img_full_name{b} =  strcat(folder_name,img_name(b).name);%img_name{b}
end
end_layer = 'prob';
input_data = {preprocess(img_full_name, batch_size)};
blob_index = net.name2blob_index('data');
net.blob_vec(blob_index).reshape([ 227 227 3 batch_size]);
net.blobs('data').set_data(input_data{1});
net.forward_prefilled();
scores = net.blobs(end_layer).get_data();   
[~, maxlabel] = max(scores);
maxlabel = maxlabel-1;

attribute = scores(2,:)';
attributes_one = [attributes_one; attribute];
    

attributes = [attributes, attributes_one];
size(attributes)
caffe.reset_all();
end

save('attributes_sunny.mat','attributes');

end


function crops_data = preprocess(img_full_name, batchsize)
crops_data = zeros(227, 227, 3, batchsize, 'single');
parfor i = 1:batchsize
    im = imread(img_full_name{i});
    if size(im, 3) == 1
        im = repmat(im,[1 1 3]);
    end
    im = imresize(im, [227,227]);
%     [h,w,c] = size(im);
    %mean_pix =  [104.0, 116.0, 122.0];
    im_data = im(:, :, [3 2 1]);  % permute channels from RGB to BGR
    im_data = single(im_data);  % convert from uint8 to single
    crops_data(:,:,:,i) = im_data;
    
%     for c = 1:3
%         crops_data(:, :, c, i) = im_data(:,:,c); %- mean_pix(c);
%     end
    
end


end




function crops_data = prepare_image(im)
% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels

IMAGE_DIM = 256;
CROPPED_DIM = 227;

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
if size(im, 3) == 1
    im = repmat(im,[1 1 3]);
end
im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
im_data = permute(im_data, [2, 1, 3]);  % flip width and height
im_data = single(im_data);  % convert from uint8 to single
im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear');  % resize im_data
% im_data = im_data- mean_data;  % subtract mean_data (already in W x H x C, BGR)

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

