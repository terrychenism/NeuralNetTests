# -*- coding: utf-8 -*-

"""
KDD 
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()

donations = pd.read_csv('donations.csv')
projects = pd.read_csv('projects.csv')
outcomes = pd.read_csv('outcomes.csv')
resources = pd.read_csv('resources.csv')
sample = pd.read_csv('sampleSubmission.csv')
essays = pd.read_csv('essays.csv')


essays = essays.sort('projectid')
projects = projects.sort('projectid')
sample = sample.sort('projectid')
ess_proj = pd.merge(essays, projects, on='projectid')
outcomes = outcomes.sort('projectid')


outcomes_arr = np.array(outcomes)


labels = outcomes_arr[:,1]

ess_proj['essay'] = ess_proj['essay'].apply(clean)

ess_proj_arr = np.array(ess_proj)

train_idx = np.where(ess_proj_arr[:,-1] < '2014-01-01')[0]
test_idx = np.where(ess_proj_arr[:,-1] >= '2014-01-01')[0]


traindata = ess_proj_arr[train_idx,:]
testdata = ess_proj_arr[test_idx,:]


tfidf = TfidfVectorizer(min_df=3,  max_features=1000)

tfidf.fit(traindata[:,5])
tr = tfidf.transform(traindata[:,5])
ts = tfidf.transform(testdata[:,5])


lr = linear_model.LogisticRegression()
lr.fit(tr, labels=='t')
preds =lr.predict_proba(ts)[:,1]


sample['is_exciting'] = preds
sample.to_csv('predictions.csv', index = False)
