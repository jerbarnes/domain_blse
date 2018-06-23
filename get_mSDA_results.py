from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np
import os
import json
import re

def open_file(file):
    f = re.sub('-1','0',open(file).read())
    return np.array(f.splitlines(), dtype=int)
    
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

def get_mSDA_results(DIR):
    sem2013_gold = open_file(os.path.join(DIR, 'semeval_2013.gold.txt'))
    sem2016_gold = open_file(os.path.join(DIR, 'semeval_2016.gold.txt'))
    books_gold = open_file(os.path.join(DIR, 'books-gold.txt'))
    dvd_gold = open_file(os.path.join(DIR, 'dvd-gold.txt'))
    electronics_gold = open_file(os.path.join(DIR, 'electronics-gold.txt'))
    kitchen_gold = open_file(os.path.join(DIR, 'kitchen-gold.txt'))

    results = {}

    for train in ['books', 'dvd', 'kitchen', 'electronics', 'all']:
        results[train] = {}
        for name, test in [('semeval_2013', sem2013_gold), ('semeval_2016', sem2016_gold),
                           ('books', books_gold), ('dvd', dvd_gold),
                           ('electronics', electronics_gold), ('kitchen', kitchen_gold)]:
            if name != train:
                try:
                    results[train][name] = {}
                    results[train][name]['mSDA'] = {}
                    pred = open_file(os.path.join(DIR, 'mSDA-{0}-{1}.txt'.format(train, name)))
                    #f1 = per_class_f1(test, pred).mean()
                    acc = accuracy_score(pred, test)
                    prec = precision_score(pred, test)
                    recall = recall_score(pred, test)
                    f1 = per_class_f1(test, pred).mean()
                    print('{0} -> {1}'.format(train, name))
                    print('{0:.3f}'.format(f1))

                    results[train][name]['mSDA']['acc'] = acc
                    results[train][name]['mSDA']['prec'] = prec
                    results[train][name]['mSDA']['rec'] = recall
                    results[train][name]['mSDA']['f1'] = f1
                except FileNotFoundError:
                    pass

    with open('results/mSDA.txt', 'w') as out:
        json.dump(results, out)

    return results


if __name__ == '__main__':
    #results = get_mSDA_results('baselines/mSDA/examples/results')
    results = get_mSDA_results('baselines/mSDA/full_semeval_results')