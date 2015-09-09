label_dir = 'E:/KITTI/training/label_2';
image_dir = 'E:/KITTI/training/image_2';
image_to_dir = 'E:/KITTI/training/image_3';
dr = dir(strcat(image_dir,'/','*.png'));

for i = 1:length(dr)
    img_name = [image_dir,'/',dr(i).name];
    %disp(img_name);
    im = imread(img_name);
    im = imresize(im, [375 1242]);
    img_to_name = [image_to_dir,'/',dr(i).name];
    imwrite(im,img_to_name)
end
