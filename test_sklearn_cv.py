import sys
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from util import get_test_train, get_score
from prprocess_data import preprocess_bank_data
#from new_encoder import preprocess_bank_data
from sklearn.cross_validation import KFold

#training_features, training_response, validate_features, validate_response, test_features, test_response = get_test_train()


def get_lg_model_score(training_features, training_response, test_features, test_response):
# Run the optimization over different values of the regularization strengh
    penalty_func = 'l2'
    C = 1
    model = LogisticRegression(penalty=penalty_func, dual=False,  tol=0.00001, C=C, fit_intercept=True)
    model.fit(training_features, training_response)
    test_pred = model.predict_proba(test_features)

    return get_score( test_response, test_pred[:,1])
 

def get_sgd_model_score(training_features, training_response, test_features, test_response):
# Run the optimization over different values of the regularization strengh
    penalty_func = 'l1'
    C = 1
    model = SGDClassifier(loss='log', penalty=penalty_func, fit_intercept=True, n_iter=500, alpha=C )
    #model = LogisticRegression(penalty=penalty_func, dual=False,  tol=0.01, C=C, fit_intercept=True)
    model.fit(training_features, training_response)
    test_pred = model.predict_proba(test_features)
    return get_score( test_response, test_pred[:,1])
 


X, y = preprocess_bank_data()
kf = KFold(y.size, n_folds=10)

auc_scores = []
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #score = get_lg_model_score(X_train, y_train, X_test, y_test)
    score = get_sgd_model_score(X_train, y_train, X_test, y_test)
    print score
    auc_scores.append(score)

mn = np.array( auc_scores)
print "AUC", mn.mean() 