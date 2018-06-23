import os, sys
from sklearn.metrics import f1_score
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse
from MyMetrics import *


def open_file(file):
    return np.array(open(file).readlines(), dtype=int)

def get_results(lang, model, bi=True):
    batch = None
    epoch = None
    alpha = None
    
    if bi:
        gold = open_file('../predictions/opener-{0}-binary.txt'.format(lang))
        preds = [os.path.join('../predictions', lang, model, f) for f in os.listdir(os.path.join('../predictions', lang, model)) if 'bi' in f]
    else:
        gold = open_file('../predictions/opener-{0}-4cls.txt'.format(lang))
        preds = [os.path.join('../predictions', lang, model, f) for f in os.listdir(os.path.join('../predictions', lang, model)) if '4cls' in f]

    results = {}
    for f in preds:
        pred = open_file(f)
        try:
            """
            if bi:
                f1 = macro_f1(gold, pred)
            else:
                f1 = f1_score(gold, pred, labels=sorted(set(gold)), average='macro')
            """
            f1 = micro_f1(gold, pred)
            if 'batch' in f:
                batch = int(re.findall('[0-9]+', f.split('-')[-1])[0])
            if 'epoch' in f:
                epoch = int(re.findall('[0-9]+', f.split('-')[-2])[0])
            if 'alpha' in f:
                alpha = float(re.findall('[0-9]\.[0-9]+', f)[0])
            results[(epoch, batch, alpha)] = f1
        except ValueError:
            pass
    return results

def get_best_results_parameters(lang, model, bi=True):
    results = get_results(lang, model, bi)
    params = list(results.keys())[np.argmax([i for i in results.values()])]
    results = results[params]
    return params, results

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
    a = [np.eye(num_classes)[i] for i in X]
    return np.array(a)

def micro_f1(y, pred):
    true_pos, false_pos, false_neg = [], [], []
    num_labels = len(set(y))
    y = to_array(y, num_labels)
    pred = to_array(pred, num_labels)
    for emo in range(num_labels):
        tp = 0
        fp = 0
        fn = 0
        for i, j in enumerate(y[:,emo]):
            if j == 1 and pred[:,emo][i] == 1:
                tp += 1
            elif j == 1 and pred[:,emo][i] == 0:
                fn += 1
            elif j == 0 and pred[:,emo][i] == 1:
                fp += 1
        true_pos.append(tp)
        false_pos.append(fp)
        false_neg.append(fn)
        
    true_pos = np.array(true_pos)
    false_pos = np.array(false_pos)
    false_neg = np.array(false_neg)
    micro_precision = true_pos.sum() / (true_pos.sum() + false_pos.sum())
    micro_recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))
    return micro_precision, micro_recall, micro_f1

def macro_f1(y, pred):
    precisions, recalls = [], []
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    for emo in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        for i, j in enumerate(y[:,emo]):
            if j == 1 and pred[:,emo][i] == 1:
                tp += 1
            elif j == 1 and pred[:,emo][i] == 0:
                fn += 1
            elif j == 0 and pred[:,emo][i] == 1:
                fp += 1
        try:
            pr = tp / (tp + fp)
        except ZeroDivisionError:
            pr = 0
        try:
            rc = tp / (tp + fn)
        except ZeroDivisionError:
            rc = 0
        precisions.append(pr)
        recalls.append(rc)
    precisions = np.array(precisions)
    recalls = np.array(precisions)
    macro_precision = precisions.mean()
    macro_recall = recalls.mean()
    macro_f1 = 2 * ((macro_precision * macro_recall) / (macro_precision + macro_recall))
    return macro_f1

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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    
    for lang in ['es', 'ca', 'eu']:
        for binary in [True, False]:
            models = []
            params = []
            results = []
            for model in ['ble', 'mt-svm', 'artetxe-svm', 'barista-svm']:
                try:
                    best_params, best_f1 = get_best_results_parameters(lang, model, binary)
                    models.append(model)
                    if model == 'ble':
                        params.append(best_params)
                    results.append(best_f1)
                except:
                    pass
            print('#### {0} {1} ####'.format(lang, binary))
            print('\t\t'.join(models))
            try:
                print(params[0])
            except:
                pass
            print('\t\t'.join(['{0:.3f}'.format(i) for i in results]))


if __name__ == '__main__':
    args = sys.argv
    main(args)
