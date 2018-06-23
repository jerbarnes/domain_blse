import os
from sklearn.metrics import f1_score
import numpy as np
import re
import matplotlib.pyplot as plt


def open_file(file):
    return np.array(open(file).readlines(), dtype=int)

def get_results(lang, model, bi=True):
    if bi:
        gold = open_file('../predictions/opener-{0}-binary.txt'.format(lang))
        preds = [os.path.join('../predictions', lang, model, f) for f in os.listdir(os.path.join('../predictions', lang, model)) if 'bi' in f]
    else:
        gold = open_file('../predictions/opener-{0}-4cls.txt'.format(lang))
        preds = [os.path.join('../predictions', lang, model, f) for f in os.listdir(os.path.join('../predictions', lang, model)) if '4cls' in f]

    results = {}
    for f in preds:
        pred = open_file(f)
        if bi:
            f1 = f1_score(gold, pred, labels=sorted(set(gold)), average='binary')
        else:
            f1 = f1_score(gold, pred, labels=sorted(set(gold)), average='macro')
        batch = re.findall('[0-9]+', f.split('-')[-1])[0]
        epoch = re.findall('[0-9]+', f.split('-')[-2])[0]
        results[(int(epoch), int(batch))] = f1
    return results

def heatmap(results):
    xs = sorted(set([i[0] for i in results.keys()]))
    ys = sorted(set([i[1] for i in results.keys()]))
    a = np.zeros((len(xs), len(ys)))
    for i,x in enumerate(xs):
        for j, y in enumerate(ys):
            try:
                a[i,j] = results[(x,y)]
            except:
                pass
    heatmap = plt.pcolor(a, cmap='Blues')
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            plt.text(x + 0.5, y + 0.5, '%.2f' % a[y, x],
                horizontalalignment='center',
                verticalalignment='center',)
    plt.grid()
    plt.colorbar(heatmap)
    plt.ylabel('Batch Size')
    plt.xlabel('Epochs')
    plt.show()

def to_array(X, num_classes):
    a = []
    for i in X:
        a.append(np.eye(num_classes)[i])
    return np.array(a)

def per_class_f1(y, pred):
    """Get the per class f1 score"""
    
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    results = []
    for j in range(num_classes):
        class_y = y[:,j]
        class_pred = pred[:,j]
        mm = MyMetrics(class_y, class_pred, one_hot=False, average='binary')
        prec, rec, f1 = mm.get_scores()
        results.append([prec, rec, f1])
    return np.array(results)
