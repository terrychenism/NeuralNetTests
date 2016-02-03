# -*- coding: utf-8 -*-
"""

Face Normalized

the normalized image size: 150*120
normalized coordinates [x,y]
eye_left:   [36,62]
eye_right:  [84,62]
nose:       [60,93]
mouth_left: [42,117]
mouth_right:[78,117]

Author: Tairui Chen
"""

import numpy as np
import os
import pickle

cv_root = 'F:/Program Files/OpenCV248/opencv/build/python/2.7/x64/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, cv_root)
import cv2

#the matrix of the normalized coordinates [y',x'] of the landmarks 
#eye_left eye_rigth nose mouth_left mouth_right
eye_left=   np.array([62,36],dtype=float)
eye_right=  np.array([62,84],dtype=float)
nose     =  np.array([93,60],dtype=float)
mouth_left= np.array([117,42],dtype=float)
moutn_right=np.array([117,78],dtype=float)

#M is the matrix composed of the normalized coordinates
m1=np.append(eye_left,[0,0,1,0])
m2=np.append(np.append([0,0],eye_left),[0,1])
M =np.append([m1],[m2],axis=0)

m1=np.append(eye_right,[0,0,1,0])
m2=np.append(np.append([0,0],eye_right),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(nose,[0,0,1,0])
m2=np.append(np.append([0,0],nose),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(mouth_left,[0,0,1,0])
m2=np.append(np.append([0,0],mouth_left),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

m1=np.append(moutn_right,[0,0,1,0])
m2=np.append(np.append([0,0],moutn_right),[0,1])
M =np.append(np.append(M,[m1],axis=0),[m2],axis=0)

#transfor M to matrix type
M =np.matrix(M)

#Row_size: is the total number of the rows of the normalized image
#Col_size: is the total number of the columns of the normalized image
Row_size=150
Col_size=120


Row_ori=512
Col_ori=314

#dir_path: the path of the folder containing the image 
dir_path='./'
#the image format in the database
img_fmt='.jpg'

#dest_dir: the path of the folder saving the normalized image
dest_dir='./'


landmark = np.matrix([[ 124.41716316,  197.3844431 ],
 [ 191.60630989,  192.90364423],
 [ 161.647524  ,  235.38231599],
 [ 134.93097377,  272.37337595],
 [ 192.42444247,  268.54918301]])



def Affain_trans(old_face,old_rgb, h,Row_size=150,Col_size=120):
    '''
    return the new_face Mat by the affain transformation
    old_face: the old_face Mat
    h: the coeffients of the affain transformation
    '''
    new_face=np.zeros((Row_size,Col_size),np.uint8)
    for y in range(Row_size):
        for x in range(Col_size):
            #xynew=[y,x]
            xynew=np.matrix([[y,x,0,0,1,0],[0,0,y,x,0,1]])*h
                        
            
            fy = int(np.floor(xynew[0]))
            cy = int(np.ceil(xynew[0]))
            ry = int(round(xynew[0]))
            fx = int(np.floor(xynew[1]))
            cx = int(np.ceil(xynew[1]))
            rx = int(round(xynew[1]))
            
            '''
            if ry<0 or rx<0 or ry>=Row_ori or rx>=Col_ori:
                new_face[y][x]=0
            else:
                new_face[y][x]=old_face[ry][rx]
            '''
            
            
            #check the interpolation needed or not
            if (abs(xynew[1]-rx)<1e-06) and (abs(xynew[0]-ry)<1e-6):
                #interpolation is not needed
                new_face[y][x]=old_face[ry][rx]

            elif fy<0 or fx<0 or cy>=Row_ori or cx>=Col_ori:
                #or fx<0 or fy<0 or cy>Row_ori or cx>Col_ori:
                new_face[y][x]=0
                
            else:
                    
                #interpolation is needed
                ty = xynew[0]-fy
                
                tx = xynew[1]-fx
            
                #Calculate the interpolation weights of the four neighbors
                w1 = (1-tx)*(1-ty)
                w2 = tx*(1-ty)
                w3 = (1-tx)*ty
                w4 = tx*ty
                new_face = old_rgb      
                for i in range(3):    
                    new_face[y][x][i]=np.uint8(w1*old_rgb[fy][fx][i]+w2*old_rgb[fy][cx][i]+\
                                    w3*old_rgb[cy][fx][i]+w4*old_rgb[cy][cx][i])
            
            
            
            
    return new_face

        
        
'''
calculate the affain transformation coeffcients h=[a b c d e f]
xy=[y1,x1,y2,x2,...y5,x5] is the coordinate of the five landmarks in the origen image
x=ax'+by'+c
y=dx'+ey'+f
xy'=Mh
'''        
# xy=np.array([Row_ori*landmark[0,1],Col_ori*landmark[0,0],\
#     Row_ori*landmark[1,1],Col_ori*landmark[1,0],\
#     Row_ori*landmark[2,1],Col_ori*landmark[2,0],\
#     Row_ori*landmark[3,1],Col_ori*landmark[3,0],\
#     Row_ori*landmark[4,1],Col_ori*landmark[4,0]])
xy = np.array([   197.3844431, 124.41716316,\
                  192.90364423,191.60630989,\
                  235.38231599,161.647524  ,\
                  272.37337595,134.93097377,\
                  268.54918301,192.42444247])

#xy=0.01*xy
xy=np.matrix(xy)
print xy
M_T=M.T
h=np.linalg.inv(M_T*M)*M_T*xy.T

# new face based on the affain transformation
img_name="025"
old_face=cv2.imread(os.path.join(dir_path, img_name+img_fmt))
old_face_gray=cv2.cvtColor(old_face,cv2.COLOR_BGR2GRAY)
new_face =Affain_trans(old_face_gray, old_face,h)

for i in range(5):
    cv2.circle(old_face, (int(landmark[i,0]), int(landmark[i,1])), 2, (0,255,0), -1)

cv2.imwrite("new_face.jpg",new_face)
# print root
cv2.imshow("sadsds", new_face)
cv2.waitKey(2000)
