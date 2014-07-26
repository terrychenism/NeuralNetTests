import sys
import numpy
from numpy import array
from scipy.optimize import leastsq


def __residual(params, xdata, ydata):#guess the function is cosine dot
    return (ydata - numpy.dot(xdata, params))

f_in = open('input.txt','r')



parameter_matrix = numpy.zeros(shape=(10,20))
y_list = list()

#read in these 8809=6
current = 0
for l in f_in.readlines():
    p,y = l.strip().split('=')
    y_list.append(float(y))
    for t in p:
        x = int(t)
        parameter_matrix[x][current]+=1
    current+=1
    
f_in.close()

ydata=numpy.array(y_list)

print 'the freq matrix is '
for i in range(current):
    s = ','.join([str(parameter_matrix[j][i]) for j in range(10)])
    print s

xdata = numpy.transpose(parameter_matrix)

x0=numpy.array([1,1,1,1,1,1,1,1,1,1])#initial guess 

print 'fitting parameter is'
print leastsq(__residual, x0, args=(xdata, ydata))
