"""
Project: Bike Sharing Demand
Date: June 3 2014
"""

dataPath = "/data/"
outPath = "results/"

import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def loadData(datafile):
    return pd.read_csv(datafile)

def splitDatetime(data):
    sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
    date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
    time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
    data['year'] = date['year']
    data['month'] = date['month']
    data['day'] = date['day']
    data['hour'] = time['hour'].astype(int)
    return data

def createDecisionTree():
    est = DecisionTreeRegressor()
    return est

def createRandomForest():
    est = RandomForestRegressor(n_estimators=500)
    return est

def createExtraTree():
    est = ExtraTreesRegressor(n_estimators=628)
    return est
    
def createGradientBoosting():
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0, loss='ls')
    return est

def createKNN():
    est = KNeighborsRegressor(n_neighbors=2)
    return est
    
def predict(est, train, test, features, target):

    est.fit(train[features], train[target])

    with open(outPath + "submission.csv", 'wb') as f:
        f.write("datetime,count\n")

        for index, value in enumerate(list(est.predict(test[features]))):
            f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))


def main():

    train = loadData("train.csv")
    test = loadData("test.csv")

    train = splitDatetime(train)
    test = splitDatetime(test)

    target = 'count'
    features = [col for col in train.columns if col not in ['datetime', 'casual', 'registered', 'count']]

    #est = createDecisionTree()
    #est = createRandomForest()
    est = createExtraTree()
    #est = createGradientBoostingRegressor()
    #est = createKNN()
    #train, test = normalize(train[features], test[features])
    predict(est, train, test, features, target)



if __name__ == "__main__":
    main()
