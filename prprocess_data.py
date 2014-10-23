#We are preprocessing to convert the categorical data into a one hot encoding scheme
# via the dictionary vectorizer.
import sys
import pandas
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from util import split_data 
from sklearn.cross_validation import KFold

def preprocess_bank_data():
    categ_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
    num_cols  = ["age", "balance", "day", "duration", "campaign","pdays","previous"]
    response = ["y"]

    data = []
    resp = []
    resp_dict = {"yes":1, "no":0}
    with open("datasets/bank/bank.csv", "r") as fh:
        hdr = fh.readline()
        cols = hdr.strip().split(";")
        count = 0
        for line in fh.readlines():
            cdict = {}
            lvals = line.strip().split(";")
            for nm, val in zip( cols, lvals):
                nm   = nm.replace("\"", "")
                if nm in num_cols:
                    cdict[nm] = int(val)
                elif nm in categ_cols:
                    cdict[nm] = val
                else:
                    #print nm, val
                    val = val.replace("\"", "")
                    resp.append( resp_dict[val])
            data.append(cdict)

    vec = DictVectorizer()
    subd = vec.fit_transform(data).toarray()
    X = pandas.DataFrame(subd)
    resparr = np.array(resp)
    print vec.get_feature_names()
    return subd, resparr 

def get_bank_data():
    features, resp = preprocess_bank_data()
    return split_data(features, resp)



if __name__ == "__main__":
    features, resp = preprocess_bank_data()
    kf = cross_validation.KFold(resp.size, n_folds=10)
    for train_index, test_index in kf:
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
 