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

def main():
    
    # collect argvs
    task = "test"
    assert task in ["train", "test"]
    # sample_submission_file = sys.argv[2]
    image_folder = "C:/Users/cht2pal/Downloads/weather/test"
    file_out = "C:/Users/cht2pal/Desktop/test_list"

    # read in header
    #header = csv.reader( file(sample_submission_file) ).next()
    #header = header[1:]

    # make image list
    img_lst = []
    cnt = 0
    # if task == "train":
    #     for i in xrange(len(header)):
    #         lst = os.listdir(image_folder + header[i])
    #         for img in lst:
    #             img_lst.append((header[i] + '/' + img, i))
    #             cnt += 1
    # else:
    lst = os.listdir(image_folder)

    cls = 0
    for folder in lst:
        imglst = os.listdir(image_folder+'/'+folder)
        
        for img in imglst:
            img_lst.append(((image_folder+'/'+folder+'/'+img), cls))
            cnt += 1
        cls += 1

    # write
    with open(file_out, "w") as f:
        writer_ = csv.writer(f, delimiter='\t', lineterminator='\n')
        for item in img_lst:
            writer_.writerow(item)


if __name__ == "__main__":
    main()
