#!/usr/bin/env python

"""
@file gen_image_list.py
@brief generate image list
@author Tairui Chen
"""
# http://stackoverflow.com/questions/2584589/search-jpeg-files-using-python

import os
import sys
import csv
from random import shuffle


def main():
    
    # collect argvs
    task = "test"
    assert task in ["train", "test"]
    image_folder = "/home/tc/Downloads/YouTubeFaces/aligned_images_DB/"
    file_out = "/home/tc/Downloads/train_list.txt"
    img_lst = []
    cnt = 0

    lst = os.listdir(image_folder)

    cls = 0
    for folder in lst:
        image_sub_folder = os.listdir(image_folder+'/'+folder)
        for sub_folder in image_sub_folder:
        	imglst = os.listdir(image_folder+'/'+folder+'/'+sub_folder)
	        for img in imglst:
	            img_lst.append(((image_folder+'/'+folder+'/'+sub_folder+'/'+img), cls))
	            cnt += 1
	        cls += 1
        cls += 1

    # write
    shuffle(img_lst)
    with open(file_out, "w") as f:
        writer_ = csv.writer(f, delimiter='\t', lineterminator='\n')
        for item in img_lst:
            writer_.writerow(item)


if __name__ == "__main__":
    main()
