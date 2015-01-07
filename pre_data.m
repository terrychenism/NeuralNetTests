%% this script used for seq03-img-left dataset cropping
%% multiple object tracking 

clear all; close all; clc

idl=readIDL('bahnhof-annot.idl');
len = size(idl,2);

for i = 1:len
    img_list(i,:) = idl(i).img;
end
idl=readIDL('refined.idl');
n = 0;
for m = 1:len
initstate = idl(m).bb;


% imgname = 'seq03-img-left/image_00000001_0.png';
imgname = img_list(m,:);
im = imread(imgname);
imshow(uint8(im));
hold on;
    for i = 1: size(initstate,1)
    
        if initstate(i,1) > initstate(i,3)
            t1 = initstate(i,1);
            initstate(i,1) = initstate(i,3);
            initstate(i,3) = t1;
        end

        if initstate(i,2) > initstate(i,4)
            t2= initstate(i,2);
            initstate(i,2) = initstate(i,4);
            initstate(i,4) = t2;
        end

initstate(i,3) = initstate(i,3) - initstate(i,1) ;
initstate(i,4) = initstate(i,4) - initstate(i,2) ;


        n = n + 1;
        filename = ['patch/img', int2str(n),'.jpg'];
        if initstate(i,3) > 0 && initstate(i,4) > 0
            rectangle('Position',initstate(i,:),'LineWidth',4,'EdgeColor','r');
        end
        I = imcrop(im, initstate(i,:));
        imwrite(I,filename);
    end
pause(0.00001); 
hold off;
end
