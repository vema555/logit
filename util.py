import sys
import pandas
import numpy as np 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score , confusion_matrix

def get_data():
	alldata = pandas.read_csv("prostate.csv", header=0, index_col=0)
	print alldata.tail().to_string()
	alldata = alldata.dropna()
	response = "CAPSULE"
	ftnames = list(alldata.columns)
	ftnames.remove(response)
	return alldata[ftnames].values, np.ravel(alldata[response].values)


def split_data(alldata, response):
	N, nc = alldata.shape
	trnN = int(N * .6)
	cvN = int(N * .8)
	training_features = alldata[:trnN, :]
	training_response = response[:trnN]
	validate_features = alldata[trnN:cvN, :]
	validate_response = response[trnN:cvN]
	test_features = alldata[cvN:]
	test_response = response[cvN:]
	return training_features, training_response, validate_features, validate_response, test_features, test_response

def get_test_train():
	xdata , y = get_data()
	return 	split_data(xdata, y)


def get_score(ytrue, ypred):
 	return roc_auc_score(ytrue, ypred )