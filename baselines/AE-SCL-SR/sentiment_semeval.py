import xml.etree.ElementTree as ET
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score, f1_score, accuracy_score
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import os
import sys
sys.path.append('../../../')
from Utils.Semeval_2013_Dataset import *

def XML2arrayRAW(neg_path, pos_path):
    reviews = []
    negReviews = []
    posReviews = []

    neg_tree = ET.parse(neg_path)
    neg_root = neg_tree.getroot()
    for rev in neg_root.iter('review'):
        reviews.append(rev.text)
        negReviews.append(rev.text)



    pos_tree = ET.parse(pos_path)
    pos_root = pos_tree.getroot()

    for rev in pos_root.iter('review'):
        reviews.append(rev.text)
        posReviews.append(rev.text)

    return reviews,negReviews,posReviews

def GetTopNMI(n,CountVectorizer,X,target):
    MI = []
    length = X.shape[1]


    for i in range(length):
        temp=mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs,MI


def getCounts(X,i):

    return (sum(X[:,i]))

def extract_and_split(neg_path, pos_path):
    reviews,n,p = XML2arrayRAW(neg_path, pos_path)
    #train, train_target, test, test_target = split_data_balanced(reviews,1000,200)
    train=reviews
    train_target=[]
    test = []
    test_target=[]
    train_target = [0]*1000+[1]*1000
    return train, train_target, test, test_target


def to_array(X, n=2):
    """
    Converts a list scalars to an array of size len(X) x n
    >>> to_array([0,1], n=2)
    >>> array([[ 1.,  0.],
               [ 0.,  1.]])
    """
    return np.array([np.eye(n)[x] for x in X])

def per_class_f1(y, pred):
    """
    Returns the per class f1 score.
    Todo: make this cleaner.
    """
    
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)


def sent(src,dest,pivot_num,pivot_min_st,dim,c_parm):
    pivotsCounts = []
    #get representation matrix

    weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st) + "_" + str(dim)+".npy"
    mat= np.load(weight_str)

    mat = mat[0]

    filename = src + "_to_" + dest + "/split/"
    if not os.path.exists(os.path.dirname(filename)):
        #gets all the train and test for sentiment classification
        train, train_target, test, test_target = extract_and_split("../data/"+src+"/negative.parsed","../data/"+src+"/positive.parsed")
    else:
        with open(src + "_to_" + dest + "/split/train", 'rb') as f:
            train = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test", 'rb') as f:
            test = pickle.load(f)
        with open(src + "_to_" + dest + "/split/train_target", 'rb') as f:
            train_target = pickle.load(f)
        with open(src + "_to_" + dest + "/split/test_target", 'rb') as f:
            test_target = pickle.load(f)

    unlabeled, source, target = XML2arrayRAW("../data/" + src + "/" + src + "UN.txt","../data/" + dest + "/" + dest + "UN.txt")
    unlabeled = source + train+ target


    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=40, binary=True)
    X_2_train_unlabeled = bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

    filename = src + "_to_" + dest + "/" + "pivotsCounts/" + "pivotsCounts" + src + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st)
    with open(filename, 'rb') as f:
        pivotsCounts = pickle.load(f)




    trainSent=train
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=20, binary=True)
    X_2_train = bigram_vectorizer.fit_transform(trainSent).toarray()
    X_2_test_unlabeld = bigram_vectorizer_unlabeled.transform(trainSent).toarray()
    XforREP = np.delete(X_2_test_unlabeld, pivotsCounts, 1)  # delete second column of C
    
    rep = XforREP.dot(mat)


    X_dev_test = bigram_vectorizer.transform(test).toarray()
    X_dev_test_unlabeled = bigram_vectorizer_unlabeled.transform(test).toarray()
    XforREP_dev = np.delete(X_dev_test_unlabeled, pivotsCounts, 1)  # delete second column of C
    XforREP_dev = XforREP_dev.dot(mat)
    devAllFeatures = np.concatenate((X_dev_test,XforREP_dev),1)




    allfeatures = np.concatenate((X_2_train, rep), axis=1)


    
    if dest == 'semeval_2013':
        semeval_dataset = Semeval_Dataset('../../../datasets/semeval_2013', None, rep=words, one_hot=False, binary=True)
        dest_test = [' '.join(s) for s in semeval_dataset._Xtest]
        dest_test_target = semeval_dataset._ytest
    elif dest == 'semeval_2016':
        semeval_dataset = Semeval_Dataset('../../../datasets/semeval_2016', None, rep=words, one_hot=False, binary=True)
        dest_test = [' '.join(s) for s in semeval_dataset._Xtest]
        dest_test_target = semeval_dataset._ytest
    else:
        dest_test, source, target = XML2arrayRAW("../data/" + dest + "/negative.parsed", "../data/"+dest+"/positive.parsed")
        dest_test_target= [0]*1000+[1]*1000
    
    X_dest = bigram_vectorizer.transform(dest_test).toarray()
    X_2_test = bigram_vectorizer_unlabeled.transform(dest_test).toarray()
    x_pivot = sum(sum(X_2_test[:,pivotsCounts]))/2000.0
    print(" the avg pivots is ",x_pivot)
    XforREP_dest = np.delete(X_2_test, pivotsCounts, 1)  # delete second column of C
    rep_for_dest = XforREP_dest.dot(mat)
    allfeaturesKitchen = np.concatenate((X_dest, rep_for_dest), axis=1)


    logreg =  LogisticRegression(C=c_parm)
    logreg.fit(allfeatures, train_target)
    lg = logreg.score(allfeaturesKitchen, dest_test_target)
    log_dev_all = logreg.score(devAllFeatures,test_target)

    pred = logreg.predict(allfeaturesKitchen)
    macro_f1 = per_class_f1(dest_test_target, pred).mean()
    micro_f1 = f1_score(dest_test_target, pred, average='binary')
    acc = accuracy_score(dest_test_target, pred)
    # print prediction
    filename = os.path.join('../results', src + "_to_" + dest + '-pivots:' + str(pivot_num) + '-pivfrq:' +  str(pivot_min_st) + '-dim:' + str(dim) + '-f1:' + '{0:.3f}'.format(macro_f1))
    with open(filename, 'w') as out:
    	for p in pred:
    		out.write('{0}\n'.format(p))


    filename = src+"_to_"+dest+"/"+"results/"+src+"_to_"+dest
    if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    sentence = "pivot num = " + str(pivot_num) + " , min freq = " + str(pivot_min_st) + " dim = " + str(
        dim) + " result = " + str(lg) + " c_parm = " + str(c_parm)
    print(sentence)
    print('acc:      {0:.3f}'.format(acc))
    print('macro_f1: {0:.3f}'.format(macro_f1))
    print('micro_f1: {0:.3f}'.format(micro_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='domain adaptation from "books, kitchen, dvd, electronics, semeval_2013, semeval_2016"')
    parser.add_argument('-tr', help="training domain (default =  books)", default='books')
    parser.add_argument('-te', help="test_domain (default =  kitchen)", default='semeval_2013')
    parser.add_argument('-dim', help='number of hidden units (default = 500)', default=500, type=int)
    parser.add_argument('-min', help='minimum frequency for pivots (default = 10)', default=10, type=int)
    parser.add_argument('-piv', help='number of pivots (default = 100)', default=100, type=int)
    parser.add_argument('-c', help='C parameter for svm (default = 0.1)', default=0.1, type=float)
    args = parser.parse_args()

    sent(args.tr, args.te, args.piv, args.min, args.dim, args.c)