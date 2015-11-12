from sys import argv
import os
import sys
import csv
from random import shuffle

filename = 'train.txt'

cnt = 1
img_lst = []
with open(filename,"r") as f:
    for line in f:
       x,y = line.split()
       img_lst.append((cnt, y, x))
       cnt += 1
# write
shuffle(img_lst)
file_out = 'train.list'
with open(file_out, "w") as f:
    writer_ = csv.writer(f, delimiter='\t', lineterminator='\n')
    for item in img_lst:
       writer_.writerow(item)


