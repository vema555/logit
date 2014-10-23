import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from util import get_test_train, get_score


#training_features, training_response, validate_features, validate_response, test_features, test_response = get_test_train()

from prprocess_data import get_bank_data
training_features, training_response, validate_features, validate_response, test_features, test_response = get_bank_data()



cvec = [ 0.0001, 0.001, .01, 0.1, 1]
cvec = [ 0.01, .1, 1, 100, 1000 ]
score_best = -100
bestC = 10

penalty_func = 'l2'

for C in cvec:
	model = SGDClassifier(loss='log', penalty=penalty_func, fit_intercept=True, n_iter=1000, alpha=C )
	model.fit(training_features, training_response)
	validate_pred = model.predict(validate_features)
	score = get_score( validate_response, validate_pred)
	print C, score
	np.set_printoptions(precision=3)
 	#print model.coef_, (model.coef_[0]**2).sum()	
 	if score_best < score:
 		bestC = C
 		score_best = score


print "==============================================" , bestC
model = SGDClassifier(loss='log', penalty=penalty_func, fit_intercept=True, n_iter=1000, alpha=bestC )
model.fit(training_features, training_response)
validate_pred = model.predict(validate_features)
print bestC, get_score( validate_response, validate_pred)


print "================================================="
test_pred = model.predict(test_features)
print "Test Auc",  get_score( test_response, test_pred)
print model.coef_, (model.coef_[0]**2).sum() 