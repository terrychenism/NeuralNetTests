import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.io import loadmat


print "Creating the trainset."
   
filename = 'train.mat' 
print "Loading", filename
data = loadmat(filename, squeeze_me=True)
X = data['XX']
y = data['yy']



print "Creating the testset."        
#testname = 'test.mat' 
#print "Loading", testname
#testdata = loadmat(testname, squeeze_me=True)
#X_test = testdata['XX']



# load in training data, directly use numpy
idx = np.genfromtxt('test.csv', dtype=None, delimiter=',', skip_header=1,usecols=(0))
data = np.genfromtxt('test.csv', delimiter=',', skip_header=1 )
X_test = data[:,1:9]
#idx = idata[:,0]
print ('finish loading from csv ')
#X_test = np.vstack(X_test)    
print "Testset:", X_test.shape  
print "data", idx[100]
     
#X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
#X_train, X_test = X[1:5000], X[5000:]
#y_train, y_test = y[1:5000], y[5000:]
X_train = X[1:,:8];
y_train = y[1:];
#clf = GradientBoostingRegressor(n_estimators=1200, alpha = 0.9,learning_rate=0.1, max_depth=3, random_state=1, subsample=0.5,loss='ls').fit(X_train, y_train)
##mse  = mean_squared_error(y_test, clf.predict(X_test)) 
#print("real value: %.4f" % y[1600])
#print("estimate value: %.4f" % clf.predict(X_train[1600,:]))
##print("MSE: %.4f" % mse) 
#print "Predicting."
#y_pred = clf.predict(X_test)


rng = np.random.RandomState(1)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

clf_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6),
                          n_estimators=1500, random_state=rng)
clf_1 = DecisionTreeRegressor(max_depth=5,random_state=rng)
from sklearn.ensemble import ExtraTreesRegressor
clf_3 = ExtraTreesRegressor(n_estimators=200, max_depth=None, min_samples_split=1, random_state=0)

clf_4 = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=1, random_state=0)
clf_4.fit(X_train, y_train)
print("real value: %.4f" % y[1600])
print("estimate value: %.4f" % clf_4.predict(X_train[1600,:]))

#from sklearn.linear_model import Lasso
#from sklearn.metrics import r2_score
#alpha = 1
#lasso = Lasso(alpha=alpha,max_iter=5000,normalize=False,tol=0.01)
#
#y_pred= lasso.fit(X_train, y_train).predict(X_test)

y_pred = clf_4.predict(X_test)



for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0
    else:
         y_pred[i] = int( y_pred[i])

print
filename_submission = "submission.csv"
print "Creating submission file", filename_submission
f = open(filename_submission, "w")
print >> f, "datetime,count"
for i in range(len(y_pred)):
   print >> f, str(idx[i])+ "," + str(y_pred[i])

f.close()
print "Done."
