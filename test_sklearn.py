import sys
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from util import get_test_train, get_score
from prprocess_data import get_bank_data

#training_features, training_response, validate_features, validate_response, test_features, test_response = get_test_train()

training_features, training_response, validate_features, validate_response, test_features, test_response = get_bank_data()

# Run the optimization over different values of the regularization strengh
score_best = -100
bestC = 1
cvec = [ 0.0001, 0.001, .01, 0.1, 1, 10, 100, 10000, 1e6]

penalty_func = 'l1'


for C in cvec:
	model = LogisticRegression(penalty=penalty_func, dual=False,  tol=0.0001, C=C, fit_intercept=True)
	model.fit(training_features, training_response)
	validate_pred = model.predict(validate_features)
	score = get_score( validate_response, validate_pred)
	print C, score
	np.set_printoptions(precision=3)
 	#print model.coef_, (model.coef_[0]**2).sum()	
 	if score_best < score:
 		bestC = C
 		score_best = score

#bestC = 1e-6
print "================================================="
model = LogisticRegression(penalty=penalty_func, dual=False,  tol=0.0001, C=bestC, fit_intercept=True)
model.fit(training_features, training_response)
validate_pred = model.predict(validate_features)
print bestC, get_score( validate_response, validate_pred)

print "================================================="
test_pred = model.predict(test_features)
print "Using C", bestC, "Test Auc",  get_score( test_response, test_pred)
print model.coef_, (model.coef_[0]**2).sum()

#get_score( test_response, test_pred)
#print model.coef_, (model.coef_[0]**2).sum()
