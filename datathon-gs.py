#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 07:20:03 2018

@author: tingyid
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import svm



auth_data = {
    'grant_type'    : 'client_credentials',
    'client_id'     : '3301d48da5964d9381d1baf9ac011e36',
    'client_secret' : '9e184cf5c71071ac7354f179e389ac83d5c74f103b6f7d3bba00189f33e9d353',
    'scope'         : 'read_product_data read_financial_data'
}

# create session instance
session = requests.Session()

# make a POST to retrieve access_token
auth_request = session.post('https://idfs.gs.com/as/token.oauth2', data = auth_data)
access_token_dict = json.loads(auth_request.text)
access_token = access_token_dict['access_token']

# update session headers
session.headers.update({'Authorization':'Bearer '+ access_token})

# get Goldman-Sachs ID
request_url = 'https://api.marquee.gs.com/v1/data/USCANFPP_MINI/coverage?limit=100'
request = session.get(url=request_url)
data = json.loads(request.text)
df = pd.DataFrame(data['results'])
gsids = df.gsid

assets = []
for index, row in df.iterrows():
    assets.append(row["gsid"])
    
payload = {
    "where": {
        "gsid": assets
    },
    "fields": [ "gsid", "ticker", "name" ],
    "limit": 1000
}
    
request_url = 'https://api.marquee.gs.com/v1/assets/data/query'
request = session.post(url=request_url, json=payload)
results = json.loads(request.text)

# get GS data on Apple
payload = {
    "startDate": "2013-03-04",
    "endDate": "2018-03-04",
    "where": {
        "ticker": ["AAPL"]
    }
}
    
request_url = 'https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query'
request = session.post(url=request_url, json=payload)
results = json.loads(request.text)
data = results['data']

financialReturns = pd.DataFrame(results['data'])
financialReturns = financialReturns.set_index('date')
financialReturns.head()

# get Stock Data from IEX¶
request_url = 'https://api.iextrading.com/1.0/stock/aapl/chart/5y'
request = session.get(url=request_url)
results = json.loads(request.text)
stockData = pd.DataFrame(results)
stockData = stockData.set_index('date')
stockData.head()

# merge GS and IEX datasets based on date
financials_labeled = pd.merge(financialReturns,stockData,how='inner', left_index=True, right_index=True)
financials_labeled.head()

# create dataframe for merged dataset -- this will be our features
columns = ['financialReturnsScore','growthScore','integratedScore','multipleScore',
                   'changeOverTime','changePercent','close','high',
                  'low','open','unadjustedVolume','volume',
                  'vwap','change']
inputs = pd.DataFrame(financials_labeled, columns=columns)
inputs.dropna(axis=0, how='any', inplace=True)

time_window = range(1,60) # select time window for prediction

#for n in time_window:
n = 1
features = inputs.iloc[::n, :] # every nth day

labels = pd.DataFrame(features['close'].pct_change())
labels['close'][0] = 0
labels.columns = ['change']
labels[labels['change'] >= 0 ] = 1
labels[labels['change'] < 0 ] = -1

# normalize features 
scaler = StandardScaler()
feats_norm = pd.DataFrame(scaler.fit_transform(features), columns = columns)

# train SVM classifier
c = 0.1 # tune parameters
X_train, X_test, y_train, y_test = train_test_split(feats_norm, labels, test_size=0.3, random_state=0)
clf = svm.SVC(kernel='rbf', C=c).fit(X_train.as_matrix(), y_train.as_matrix())
print(clf.score(X_test, y_test))
ypred = clf.predict(X_test)
print(ypred)

clf = svm.SVC(kernel='rbf', C=c)
scores = cross_val_score(clf, feats_norm, labels, cv=5)
validation_score = np.mean(scores)

print(validation_score)

