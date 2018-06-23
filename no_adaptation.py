import numpy as np
from Utils.WordVecs import *
from Utils.Datasets import *
from Utils.Representations import *
from Utils.Semeval_2013_Dataset import *
from sklearn.metrics import log_loss, f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from copy import deepcopy
import json

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

def metrics(gold, pred):
    acc = accuracy_score(gold, pred)
    prec = precision_score(gold, pred)
    rec = recall_score(gold, pred)
    f1 = per_class_f1(gold, pred).mean()
    return acc, prec, rec, f1

def crossvalidation(dataset):
    best_c = 0
    best_f1 = 0
    for c in [0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1,3,6,10,30,60]:
        clf = LinearSVC(C=c)
        clf.fit(dataset._Xtrain, dataset._ytrain)
        pred = clf.predict(dataset._Xdev)
        f1 = per_class_f1(dataset._ydev, pred).mean()
        if f1 > best_f1:
            best_f1 = f1
            best_c = c
    return best_c, best_f1

if __name__ == '__main__':

    embeddingdir = '/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS'
    amazon_vecs = WordVecs(os.path.join(embeddingdir, 'SubjQuant/amazon-sg-300.txt'))
    twitter_vecs = WordVecs(os.path.join(embeddingdir, 'twitter_embeddings.txt'))

    pdataset = ProjectionDataset('lexicons/general_vocab.txt', amazon_vecs, twitter_vecs)

    books = Book_Dataset(amazon_vecs, rep=ave_vecs, one_hot=False, binary=True)
    dvd = DVD_Dataset(amazon_vecs, rep=ave_vecs, one_hot=False, binary=True)
    electronics = Electronics_Dataset(amazon_vecs, rep=ave_vecs, binary=True, one_hot=False)
    kitchen = Kitchen_Dataset(amazon_vecs, rep=ave_vecs, binary=True, one_hot=False)

    alltrain = deepcopy(kitchen)

    allX_train = [d._Xtrain for d in [books, dvd, electronics, kitchen]]
    alltrain._Xtrain =  np.array([w for l in allX_train for w in l])
    ally_train = [d._ytrain for d in [books, dvd, electronics, kitchen]]
    alltrain._ytrain = np.array([w for l in ally_train for w in l])

    allX_dev = [d._Xdev for d in [books, dvd, electronics, kitchen]]
    alltrain._Xdev =  np.array([w for l in allX_dev for w in l])
    ally_dev = [d._ydev for d in [books, dvd, electronics, kitchen]]
    alltrain._ydev = np.array([w for l in ally_dev for w in l])

    allX_test = [d._Xtest for d in [books, dvd, electronics, kitchen]]
    alltrain._Xtest =  np.array([w for l in allX_test for w in l])
    ally_test = [d._ytest for d in [books, dvd, electronics, kitchen]]
    alltrain._ytest = np.array([w for l in ally_test for w in l])



    semeval_2013 = Semeval_Dataset('datasets/semeval_2013/', amazon_vecs,
                                     binary=True, rep=ave_vecs,
                                     one_hot=False)

    semeval_2016 = Semeval_Dataset('datasets/semeval_2016/', amazon_vecs,
                                     binary=True, rep=ave_vecs,
                                     one_hot=False)

    datasets = [('books', books),
                ('dvd', dvd),
                ('electronics', electronics),
                ('kitchen', kitchen),
                ('all', alltrain),
                ('semeval_2013', semeval_2013),
                ('semeval_2016', semeval_2016)]


    results = {}

    for tr_name, train in datasets:
        results[tr_name] = {}
        for name, test in datasets:
            if not (tr_name == 'semeval_2013' and name == 'semeval_2013') and not (tr_name == 'semeval_2016' and name == 'semeval_2016'):
                results[tr_name][name] = {}
                results[tr_name][name]['none'] = {}

                best_c, best_f1 = crossvalidation(train)
                clf = LinearSVC(C=best_c)
                clf.fit(train._Xtrain, train._ytrain)
                pred = clf.predict(test._Xtest)
                acc, prec, recall, _ = metrics(test._ytest, pred)
                f1 = per_class_f1(test._ytest, pred)

                print('{0}-{1}'.format(tr_name, name))
                print(f1.mean())
                print()
                results[tr_name][name]['none']['c'] = best_c
                results[tr_name][name]['none']['dev_f1'] = best_f1
                results[tr_name][name]['none']['acc'] = acc
                results[tr_name][name]['none']['prec'] = prec
                results[tr_name][name]['none']['rec'] = recall
                results[tr_name][name]['none']['f1'] = f1.mean()

# Mono domains
semeval_2013 = Semeval_Dataset('datasets/semeval_2013/', twitter_vecs,
                                binary=True, rep=ave_vecs,
                                one_hot=False)

semeval_2016 = Semeval_Dataset('datasets/semeval_2016/', twitter_vecs,
                                binary=True, rep=ave_vecs,
                                one_hot=False)

for name, dataset in [('semeval_2013', semeval_2013),
                      ('semeval_2016', semeval_2016)]:
    results[name][name] = {}
    results[name][name]['none'] = {}
    best_c, best_f1 = crossvalidation(dataset)
    clf = LinearSVC(C=best_c)
    clf.fit(dataset._Xtrain, dataset._ytrain)
    pred = clf.predict(dataset._Xtest)
    acc, prec, recall, _ = metrics(dataset._ytest, pred)
    f1 = per_class_f1(dataset._ytest, pred)
    print('{0}-{1}'.format(name, name))
    print(f1.mean())
    print()
    results[name][name]['none']['c'] = best_c
    results[name][name]['none']['dev_f1'] = best_f1
    results[name][name]['none']['acc'] = acc
    results[name][name]['none']['prec'] = prec
    results[name][name]['none']['rec'] = recall
    results[name][name]['none']['f1'] = f1.mean()


    with open('results/no-adaptation.txt', 'w') as out:
        json.dump(results, out)