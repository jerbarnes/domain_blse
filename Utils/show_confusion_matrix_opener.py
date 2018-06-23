import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import argparse

def open_file(file):
    return np.array(open(file).readlines(), dtype=int)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def change_labels(array, binary=True):
	if binary:
		dic = {1:'Positive', 0:'Negative'}
	else:
		dic = {1:'Negative', 2:'Positive',
	           3:'Strong Positive', 0:'Strong Negative'}
	new = []
	for i in array:
		new.append(dic[i])
	return new


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', help='Gold labels')
    parser.add_argument('-p', help='Predicted labels')
    parser.add_argument('-bi', default=True, type=str2bool)

    args = vars(parser.parse_args())
    gold_file = args['g']
    pred_file = args['p']
    bi = args['bi']

    if bi:
    	labels = ['Positive', 'Negative']
    else:
    	labels = ['Strong Positive', 'Positive', 'Negative', 'Strong Negative']

    gold = change_labels(open_file(gold_file), bi)
    pred = change_labels(open_file(pred_file), bi)
    
    cm = confusion_matrix(gold, pred, labels=labels)
    plot_confusion_matrix(cm, labels)
    plt.show()

if __name__ == '__main__':

    args = sys.argv
    main(args)
