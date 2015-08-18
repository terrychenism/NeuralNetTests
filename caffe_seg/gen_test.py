import os
import sys
import csv
import pandas as pd
from sklearn import preprocessing

if len(sys.argv) < 2:
    print "Usage: python gen_test.py input_folder output_folder"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]

cmd = "convert -colorspace gray -resize 56x56^! "

img_lst = []

imgs = os.listdir(fi)

# resize original images
print "Resizing images"
for img in imgs:
    basename = os.path.basename(img)
    filename = os.path.splitext(basename)
    img_without_ext = filename[0]
    outFileName = str(img_without_ext) + ".jpg"
    md = ""
    md += cmd
    md += fi + str(img)
    md += " " + fo + outFileName
    img_lst.append((outFileName, 0))
    os.system(md)


fo = csv.writer(open("test.lst", "w"), delimiter='\t', lineterminator='\n')
for item in img_lst:
    fo.writerow(item)




==========================================
image_path = 'C:\Users\cht2pal\Downloads\weather_database/sunny/';
image_list = dir(strcat(image_path,'*.jpg'));
save_path = 'C:\Users\cht2pal\Downloads\weather_database/sunny_aug/';
for i = 1:size(image_list)
    image_name = strcat(image_path,image_list(i).name);
    im = imread(image_name);
    im = imresize(im,[256 256]);
    imwrite(im, strcat(save_path, image_list(i).name));
end
