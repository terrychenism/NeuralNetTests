"""DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

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
    #print beg, end

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
    #pl.subplot(3,1,1)
    #pl.plot(t,ynew, label='Filtered signal (%g Hz)' % f0)
    #pl.xlabel('time (seconds)')
    #n = len(ynew)
    #Ff = np.fft.rfft(ynew)
    #k = np.arange(n)
    #frq = k/1.5
    #frq = frq[range(n/2+1)]
    #pl.subplot(3,1,2)
    #print np.abs(Ff).shape + frq.shape
    #pl.plot(frq,np.abs(Ff),'r')
    #pl.xlabel('freq (Hz)')
    #pl.subplot(3,1,3)
    #pl.plot(frq,np.angle(Ff),'r')
    #pl.xlabel('freq (Hz)')

def plot_freq_y(XX, yy, t1, t2, sfreq, pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    #print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag0 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle0 = np.zeros(int((n/2)+1),dtype=float)
    Ffmag1 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle1 = np.zeros(int((n/2)+1),dtype=float)
    count = 0

    XX0 = XX[np.where(yy==0)]
    XX1 = XX[np.where(yy==1)]

    for dimX in XX0:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag0 += np.abs(f)
            Ffangle0 += np.angle(f)
            count +=1
           
    Ffmag0 = Ffmag0/count
    Ffangle0 = Ffangle0/count
   
    count = 0
    for dimX in XX1:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag1 += np.abs(f)
            Ffangle1 += np.angle(f)
            count +=1
           
    Ffmag1 = Ffmag1/count
    Ffangle1 = Ffangle1/count


    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]

    pl.subplot(2,1,1)
    pl.plot(frq,Ffmag0,'r')
    pl.plot(frq,Ffmag1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle0,'r')
    pl.plot(frq,Ffangle1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')




 
def plot_freq(XX,t1,t2,sfreq,pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag = np.zeros(int((n/2)+1),dtype=float)
    Ffangle = np.zeros(int((n/2)+1),dtype=float)
    count = 0
    for dimX in XX:
        for dimY in dimX:
            y = dimY[beg:end]
            #n = len(y)
            #print y.shape + t.shape
            f = np.fft.rfft(y)
            Ffmag += np.abs(f)
            Ffangle += np.angle(f)
            count +=1
 
    Ffmag = Ffmag/count
    Ffangle = Ffangle/count
    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]
    pl.subplot(2,1,1)
    #print np.abs(Ff).shape + frq.shape
    pl.plot(frq,Ffmag,'r')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')

    #pl.subplot(3,1,1)
    #pl.plot(t,y)
    #pl.xlabel('time (seconds)')



    return 0;

def cross_validate(clf, tr, te):
    scores = cross_validation.cross_val_score(clf, tr, te, cv=5 ) 
    print ("Cross Validation scores: mean=%f dev=%f" % (scores.mean(),scores.std()) )


def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
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
    algo = 'ksvm'
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
         mode = int(arg)
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
    subjects_train = range(1, 17) # use range(1, 17) for all subjects
    #print "Training on subjects", subjects_train 
    subjects_val = range(4, 6)

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

	#plot_freq_y(XX, yy, -0.5, 1.0, sfreq, 1)

        XX = create_features(XX, tmin, tmax, sfreq)
        #pl.show()

        #print "XX:", XX.shape
        X_train.append(XX)
        y_train.append(yy)

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


    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    print "XX:", X_train.shape
   
    #In case of tranfer learning trials from every subject should either be only in training or validation. Any kind of leakage overestimates accuracy
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_subj, y_subj, test_size=0.25, random_state=0)

    if (mode==1): #model selection
        print("Doing model selection\n")
        scores = ['accuracy','roc_auc']
        if(algo == 'l1'):
            for score in scores:
                print("Tuning hyperparameter space for metric %s " % score)
                param_grid = {'C':[1 , 10, 100, 1000, 10000]}
                gcv = GridSearchCV(estimator=LogisticRegression(C=1.0,penalty='l1',random_state=0, tol=1e-4), param_grid = param_grid, cv=5, scoring=score)
                gcv.fit(X_train, y_train)
                print()
                print(gcv.best_estimator_)
                print()
                for params, mean_score, scores in gcv.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() / 2, params))
                print()
                y_pred = gcv.predict(X_val)
                print "Accuracy score: " , metrics.accuracy_score(y_val, y_pred, normalize=True)
        elif(algo == 'l2'):
            #ls = LassoCV(n_alphas=5, cv=5)
            #ls.fit(X_train, y_train)
            #print ls.alpha_, ls.coef_
            for score in scores:
                print("Tuning hyperparameter space for metric %s " % score)
                param_grid = {'C':[.000001, .00001, .0001, .001, .01]}
                gcv = GridSearchCV(estimator=LogisticRegression(C=1.0,penalty='l2',random_state=0, tol=1e-4), param_grid = param_grid, cv=5, scoring=score)
                gcv.fit(X_train, y_train)
                print()
                print(gcv.best_estimator_)
                print()
                for params, mean_score, scores in gcv.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() / 2, params))
                print()
                y_pred = gcv.predict(X_val)
                print "Accuracy score: " , metrics.accuracy_score(y_val, y_pred, normalize=True)
        elif (algo == 'lsvm' or algo == 'ksvm'):
            tuned_params = [{'kernel':['linear'], 'C': [1, 10, 1000, 10000]},
                    {'kernel':['rbf'], 'gamma':[1e-3, 1e-4], 'C': [1,10,1000,10000]}]
            for score in scores:
                print("Tuning hyperparameter space for metric %s " % score)
                gcv = GridSearchCV(svm.SVC(C=1), tuned_params, cv=5, scoring=score)
                gcv.fit(X_train, y_train)
                print()
                print(gcv.best_estimator_)
                print()
                for params, mean_score, scores in gcv.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r"
                          % (mean_score, scores.std() / 2, params))
                print()
                y_pred = gcv.predict(X_val)
                print "Accuracy score: " , metrics.accuracy_score(y_val, y_pred, normalize=True)
    elif (mode==0): #run a specific model   
        print("Evaluating a model %s\n" % algo)

        #clf = LogisticRegression(C=10,penalty='l1',random_state=0)
        if(algo=='l1'): #C=10000
            clf = LogisticRegression(C=reg,penalty='l1',random_state=0)
        elif(algo=='l2'):
            clf = LogisticRegression(C=reg,penalty='l2',random_state=0)
            #clf = Lasso(alpha=reg, copy_X=True) 
        elif(algo=='lsvm'): #1000
            clf = svm.SVC(kernel='linear', C=reg)
        elif(algo=='ksvm'): #1000
            clf = svm.SVC(kernel='rbf', C=reg, gamma=1e-5)
        clf.fit(X_train, y_train)
        print "Classifier:"
        print clf, clf.get_params()


        #Cross validation score calculation with the selected model

        #cross_validate(clf, X_train, y_train)

        print "Training."
        #clf.fit(X_train, y_train)
        #print "Validation set score: " , clf.score(X_val, y_val)

        #print (clf.coef_.ravel()!=0).shape

        #if(algo=='l1' or algo=='l2'):
        #    coefs_ = []
        #    coefs_.append(clf.coef_.ravel().copy())
        #    pl.figure(1)
        #    pl.plot(clf.coef_.ravel())	 
        #    pl.axis('tight')
        #    pl.savefig(plotfile, bbox_inches='tight')
        #    #pl.show()

        if(fe==1): #L1 norm based feature elimination
            if(algo=='l1' or algo=='l2'):
                clf_fe = LogisticRegression(C=10000,penalty='l1',random_state=0)
                clf_fe.fit(X_train, y_train)
                X_train = X_train[:,clf_fe.coef_.ravel()!=0]
                print "Xtrain.shape: ", X_train.shape
                X_val = X_val[:,clf_fe.coef_.ravel()!=0]
                clf2 = LogisticRegression(C=reg,penalty='l2',random_state=0)
                clf2.fit(X_train, y_train)
                print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)
            elif(algo=='lsvm'):
                #clf_fe = LinearSVC(C=1000, penalty='l1',dual=False)
                #X_train = clf_fe.fit_transform(X_train,y_train)
                #print "Xtrain.shape: ", X_train.shape
                #X_val = clf_fe.transform(X_val) 
                clf_fe = LogisticRegression(C=10000,penalty='l1',random_state=0)
                clf_fe.fit(X_train, y_train)
                X_train = X_train[:,clf_fe.coef_.ravel()!=0]
                print "Xtrain.shape: ", X_train.shape
                X_val = X_val[:,clf_fe.coef_.ravel()!=0]
                clf2 = svm.SVC(kernel='linear', C=reg)
                clf2.fit(X_train, y_train)
                print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)
            elif(algo=='ksvm'):
                #clf_fe = LinearSVC(C=10000, penalty='l1',dual=False)
                #X_train = clf_fe.fit_transform(X_train,y_train)
                #print "Xtrain.shape: ", X_train.shape
                #X_val = clf_fe.transform(X_val) 
                clf2 = svm.SVC(kernel='rbf', C=reg, gamma=1e-5)
                clf2.fit(X_train, y_train)
                print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)
        elif(fe==2): #Recursive feature elimination
            #clf_fe = LogisticRegression(C=.001,penalty='l2',random_state=0)
            rfecv = RFECV(estimator=clf, step=10, cv=3, scoring='accuracy')
            rfecv.fit(X_train, y_train)
            print("Optiimal number of features : %d" % rfecv.n_features_)
            pl.figure(2)
            pl.xlabel("Number of features selected")
            pl.ylabel("Cross validation scores")
            pl.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            pl.show()
        elif(fe==3): #Randomized Lasso/ Stability Selection
            clf_fe = RandomizedLogisticRegression(C=10000, random_state=144, n_resampling=20, n_jobs=2).fit(X_train, y_train)
            X_train = clf_fe.transform(X_train)
            print "Xtrain.shape: ", X_train.shape
            X_val = clf_fe.transform(X_val)
	    if(algo=='l1' or algo=='l2'):
                clf2 = LogisticRegression(C=reg,penalty=algo,random_state=0)
            elif(algo=='lsvm'):
                clf2 = svm.SVC(kernel='linear', C=reg)
            elif(algo=='ksvm'):
		clf2 = svm.SVC(kernel='rbf', C=reg, gamma=1e-5)
            clf2.fit(X_train, y_train)
            print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)
        elif(fe==4): #Univariate feature selection F-score based
            selector = SelectPercentile(f_classif, percentile=80)
            selector.fit(X_train, y_train)
	    scores = -np.log10(selector.pvalues_)
	    scores /= scores.max()
            #pl.figure(2)
            #pl.xlabel("Features")
            #pl.ylabel("F-scores")
	    #pl.bar(scores, width=.2, label=r'Univariate score ($-Log(p_{value})$)', color='g')
            print "Xtrain.shape: ", X_train.shape
            X_train = selector.transform(X_train)
            X_val = selector.transform(X_val)
	    if(algo=='l1' or algo=='l2'):
                clf2 = LogisticRegression(C=reg,penalty=algo,random_state=0)
            elif(algo=='lsvm'):
                clf2 = svm.SVC(kernel='linear', C=reg)
            elif(algo=='ksvm'):
		clf2 = svm.SVC(kernel='rbf', C=reg, gamma=1e-5)
            clf2.fit(X_train, y_train)
            print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)
        elif(fe==5):  #Tree based feature selection
            forest = ExtraTreesClassifier(n_estimators=20, random_state=144)
            forest.fit(X_train, y_train)
	    importances = forest.feature_importances_
            std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
            pl.figure(2)
            pl.bar(range(10), importances, color="r", yerr=std, align="center")
            X_train = forest.transform(X_train, threshold=mean)
            X_val = forest.transform(X_val,threshold=mean)
	    if(algo=='l1' or algo=='l2'):
                clf2 = LogisticRegression(C=reg,penalty=algo,random_state=0)
            elif(algo=='lsvm'):
                clf2 = svm.SVC(kernel='linear', C=reg)
            elif(algo=='ksvm'):
		clf2 = svm.SVC(kernel='rbf', C=reg, gamma=1e-5)
            clf2.fit(X_train, y_train)
            print "Validation set score filtered coeff: " , clf2.score(X_val, y_val)

    pl.show()

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
    
