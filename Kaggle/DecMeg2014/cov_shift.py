"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

import math
import numpy as np
import pylab as pl
import sys, getopt
from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, RandomizedLogisticRegression, lasso_stability_path, LassoLarsCV
from sklearn import cross_validation, metrics
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV,RFE,SelectPercentile, f_classif
from sklearn.grid_search import GridSearchCV
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split as tts
from sklearn.cross_decomposition import PLSRegression

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype='bandstop')
    return(b,a)

def butter_bandpass_filter(data,lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a,data)
    return y

def filter_noise(XX,t1,t2,sfreq):
    #t = np.linspace(t1,t2,1.5/375,endpoint=False)
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)

    b, a = butter_bandpass(49, 51, sfreq, order=5)
    b1, a1 = butter_bandpass(99, 101, sfreq, order=5)

    f0 = 50
    XXnew = XX
    count =0
    for dimX in XXnew:
        for dimY in dimX:
            y = dimY[beg:end]
            #ynew = butter_bandpass_filter(y, 45, 55, sfreq, order=5)
            ynew = lfilter(b, a, y)
            ynew2 = lfilter(b1, a1, ynew)
            dimY[beg:end] = ynew2
            #print count
	    count += 1 
    return XXnew         

def cross_validate(clf, tr, te):
    scores = cross_validation.cross_val_score(clf, tr, te, cv=5 ) 
    print ("Cross Validation scores: mean=%f dev=%f" % (scores.mean(),scores.std()) )


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    #print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    #print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    #print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':


    print "Parsing arguments"
    plotfile = ''
    algo = 'log'
    reg = 1
    mode = 0
    fe=-1
    print sys.argv[1:]
    resfile = 'submission.csv'  
 
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hm:a:p:r:c:f:",["mode","algo=","pfile=","reg=","fe="])
    except getopt.GetoptError:
      print 'dhruv.py -o <plotfile>'
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print 'dhruv.py -o <plotfile> -m  mode(1=modelselection 0=specific model) -C <regularization> -a <algorithm(l1|l2|lsvm|ksvm) -f <feature extraction 0=None 1=l1 based 2=chi2>'
         sys.exit()
      elif opt in ("-m", "--mode"):  #mode= 0 clf.model 1 model selection
         mode = float(arg)
      elif opt in ("-a", "--algo"):
         algo = arg
      elif opt in ("-p", "--pfile"):
         plotfile = arg
      elif opt in ("-r", "--rfile"):
         resfile = arg
      elif opt in ("-c", "--reg"):
         reg = float(arg)
      elif opt in ("-f", "--fe"):
         fe = int(arg)
    print 'Algorithm is ', algo
    print 'Plot file is ', plotfile
    print 'Regularization Parameter is ', reg
    

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    #subjects_train = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16] # use range(1, 17) for all subjects
    subjects_train = range(1, 2)
    #print "Training on subjects", subjects_train 
    subjects_val = range(4, 5)

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0
    tmax = 0.5
    print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    X_val = []
    y_val = []

    subjId_train = []
    subjId_val = []

    print
    #print "Creating the trainset."
    for subject in subjects_train:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        #print "Dataset summary:"
        #print "XX:", XX.shape
        #print "yy:", yy.shape
        #print "sfreq:", sfreq

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        y_train.append(yy)
	subjId_train.append(subject)
        print "X_train:", len(X_train)

    for subject in subjects_val:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading val", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_val.append(XX)
        y_val.append(yy)
	subjId_val.append(subject)

    n_components = 75 #X_train[0].shape[1]

    for vid,Xt,yt in zip(subjId_val, X_val, y_val):
	levelOneTest = []
	levelOneTrain = []
	X_levelOne = []
	y_levelOne = []	
	level0Classifier = []
        for tid,Xp,yp in zip(subjId_train,X_train,y_train):
	    print "Predicting subject ", vid, "from subject ", tid
            y0 = np.zeros(yp.shape)
	    y1 = np.ones(Xt.shape[0])
	    X = np.vstack([Xp,Xt])
            yd = np.concatenate([y0,y1])

            pls = PLSRegression(n_components)
	    Xp_t, Xp_v, yp_t, yp_v = tts(Xp.copy(),yp.copy(),train_size=0.9)
	    yp_t = yp_t.astype(bool)
	    yp_t_not =  np.vstack((yp_t,~yp_t)).T
	    #print "yp_t_not ", yp_t_not.shape
	    pls.fit(Xp_t,yp_t_not.astype(int))
	    yp_new = pls.predict(Xp_t, copy=True)
	    yp_pred = (yp_new[:,0] > yp_new[:,1]).astype(int)
	    yp_t = yp_t.astype(int)
	    #print y_new,y_pred, y_t
	    error = ((yp_t - yp_pred) ** 2).sum()
   	    print "PLS Training error " , float(error)/yp_t.shape[0]
 	    yp_new = pls.predict(Xp_v, copy=True)
	    yp_pred = (yp_new[:,0] > yp_new[:,1]).astype(int)
	    #print y_new, y_pred, y_v
	    #print ((y_v - y_pred) ** 2).sum(), y_v.shape[0]
	    error = ((yp_v - yp_pred) ** 2).sum()
	    print "PLS Validation error " , float(error)/yp_v.shape[0]

	    X_new = pls.transform(X)
	    rf = RandomForestClassifier(n_estimators=500, max_depth=None, max_features=int(math.sqrt(n_components)), min_samples_split=100, random_state=144, n_jobs=4)
	    #print "shapes ", X_new.shape, y.shape
	    #print X_new,y
            X_t, X_v, y_t, y_v = tts(X_new,yd,train_size=0.85)

	    rf.fit(X_t, y_t)
            print "Random Forest Classifier: ", rf.get_params()
	    print "Covariance Classifier Training score: ", rf.score(X_t, y_t)
	    print "Covariance Classifier Validation score: ", rf.score(X_v, y_v)
	    #print "Class prob: ", zip(rf.predict_proba(X_v), y_v)

            sample_weights = rf.predict_proba(pls.transform(Xp_t))[:,1]
	    print sample_weights.shape
	    sample_weights = abs(sample_weights-0.5)

	    for a in [.01, .1, .3, 1, 3, 10, 20, 30, 40, 50, 100]:
                clf = SGDClassifier(alpha=a,loss=algo,n_iter=20) 
	        clf.fit(Xp_t,yp_t,sample_weight=sample_weights)
                clf2 = SGDClassifier(alpha=a,loss=algo,n_iter=20) 
	        clf2.fit(Xp_t,yp_t)
		print "alpha: ", a
	        print "Target score with weights: ", clf.score(Xt,yt)
	        print "Target score without weights: ", clf2.score(Xt,yt)

	    #We use the same validation data to build the level 1 classifier
	    a = -1
	    if algo == 'hinge':
	        a = 20
	    else:
		a = 10
            clf = SGDClassifier(alpha=a,loss=algo,n_iter=20) 
	    clf.fit(Xp_t,yp_t,sample_weight=sample_weights)
            level0Classifier.append(clf)
	    X_levelOne.append(Xp_v)
	    y_levelOne.append(yp_v.T)

        #print y_levelOne, y_levelOne[0].shape, y_levelOne[1].shape

	X_levelOne = np.vstack(X_levelOne)
	y_levelOne = np.concatenate(y_levelOne)

	print "Shapes of Level one X and y ", X_levelOne.shape, y_levelOne.shape
	for c in level0Classifier:
	    if(algo=='log'):
	        levelOneTrain.append(c.predict_proba(X_levelOne)[:,0].T)
	        levelOneTest.append(c.predict_proba(Xt)[:,0].T)
	    else:
	        levelOneTrain.append(c.predict(X_levelOne).T)
	        levelOneTest.append(c.predict(Xt).T)

        #print levelOneTrain, levelOneTrain[0].shape, levelOneTrain[1].shape
	    
	levelOneTrain = np.vstack(levelOneTrain).T
        levelOneTest = np.vstack(levelOneTest).T
	print "Shape of Level One test pred ", levelOneTest.shape 
	print "Shape of Level One  train predictions ", levelOneTrain.shape
	print "Shape of Level One Y ", y_levelOne.shape

	print "Level 1 classifier for subject ", vid
	for a in [.00001, .0001, .0003, .001, .01, .1, .3, .5, .8, 1]:
            s_clf = SGDClassifier(alpha=a,loss=algo, n_iter=20)
	    s_clf.fit(levelOneTrain, y_levelOne)
	
            print "alpha: ", a
	    print "Final Score " , s_clf.score(levelOneTest, yt)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    #X_val = np.vstack(X_val)
    #y_val = np.concatenate(y_val)

    print "Calculating pooled scores "
    for a in [.0001, .0003, .001, .01, .1, .3, .5]:
        pool_clf = SGDClassifier(alpha=a,loss=algo, n_iter=20)
        pool_clf.fit(X_train,y_train)
        print "alpha: ", a
        for vid,Xt,yt in zip(subjId_val, X_val, y_val):
            print "Pooled Score for subject ", vid , " : ", pool_clf.score(Xt, yt)

    if(resfile != ''):
        print "Creating the testset."
        subjects_test = range(17, 24)
        for subject in subjects_test:
            filename = 'data/test_subject%02d.mat' % subject
            print "Loading", filename
            data = loadmat(filename, squeeze_me=True)
            XX = data['X']
            ids = data['Id']
            sfreq = data['sfreq']
            tmin_original = data['tmin']
            print "Dataset summary:"
            print "XX:", XX.shape
            print "ids:", ids.shape
            print "sfreq:", sfreq
            XX = filter_noise(XX, -0.5, 1.0, sfreq)
            XX = create_features(XX, tmin, tmax, sfreq)
            X_test.append(XX)
            ids_test.append(ids)

        X_test = np.vstack(X_test)
        ids_test = np.concatenate(ids_test)
        print "Testset:", X_test.shape
                      
        print "Predicting."
        y_pred = clf.predict(X_test)

        filename_submission = "submission.csv"
        print "Creating submission file", resfile
        f = open(resfile, "w")
        print >> f, "Id,Prediction"
        for i in range(len(y_pred)):
            print >> f, str(ids_test[i]) + "," + str(y_pred[i])
        f.close()
    print "Done."
    
