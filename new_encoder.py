#We are preprocessing to convert the categorical data into a one hot encoding scheme
# via the dictionary vectorizer.
import sys
import pandas
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from util import split_data 
from sklearn.cross_validation import KFold

def find_num_varbs():
    categ_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome", "y"]
    num_cols  = ["age", "balance", "day", "duration", "campaign","pdays","previous"]
    response = ["y"]

    data = []
    resp = []
    resp_dict = {"yes":1, "no":0}
   
    with open("datasets/bank/bank.csv", "r") as fh:
        hdr = fh.readline()
        cols = hdr.strip().split(";")
        count = 0
        cdict = {}

        for line in fh.readlines():
            lvals = line.strip().split(";")
            for nm, val in zip( cols, lvals):
                nm   = nm.replace("\"", "")
                if nm in categ_cols:
                    if not cdict.has_key(nm):
                        cdict[nm] = set()
                    cdict[nm].add( val )
    encoder = {}
    for feature, val in cdict.iteritems():
        encoder[feature] = {}
        lkeys = list(val)
        N = len(val) - 1

        for i, featureval in enumerate(lkeys):
            code = np.zeros(N)
            
            if i != N:
                code[i] = 1
            encoder[feature][featureval] = code
            #print feature, featureval, i, N, code
    return encoder

def preprocess_bank_data():
    categ_cols = ["job","marital","education","default","housing","loan","contact","month","poutcome"]
    num_cols  = ["age", "balance", "day", "duration", "campaign","pdays","previous"]
    response = ["y"]

    encoder = find_num_varbs()
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
            arr = []
            for nm, val in zip( cols, lvals):
                nm   = nm.replace("\"", "")
                if nm in num_cols:
                    arr.append( int(val) )
                elif nm in categ_cols:
                    code = encoder[nm][val]
                    arr.extend(code)
                else:
                    #print nm, val
                    val = val.replace("\"", "")
                    arr.append( resp_dict[val])
                    #resp.append( resp_dict[val])

            #print ', '.join(map(str, arr))

            data.append(arr)
    hdr = []
    print cols
    for  c_ in  cols:
        c = c_.replace("\"", "")
        if c in categ_cols:
            v = ["%s_%d"%(c,i) for i in xrange(len(encoder[c])-1)]
            hdr.extend(v)
        else:
            hdr.append(c)
    
    #print hdr
    pdf = pandas.DataFrame(data, columns = hdr)
    #print pdf.tail()
    nms  = list(pdf.columns.values)
    nms.remove("y")
    subdf = pdf[nms].values
    resparr = pdf["y"].values
    return subdf, resparr 


if __name__ == "__main__":
    #find_num_varbs()
    features, response = preprocess_bank_data()
    print features.shape
    print len(response)    
    #features, resp = preprocess_bank_data()
    #kf = cross_validation.KFold(resp.size, n_folds=10)
    #for train_index, test_index in kf:
    #   print("TRAIN:", train_index, "TEST:", test_index)
    #   X_train, X_test = X[train_index], X[test_index]
    #   y_train, y_test = y[train_index], y[test_index]
   