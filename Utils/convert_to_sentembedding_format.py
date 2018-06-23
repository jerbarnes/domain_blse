import nltk
import sys
from copy import deepcopy
sys.path.append('../')
from Utils.Datasets import *
from Semeval_2013_Dataset import *

def writeout(X,Y, outfile):
    with open(outfile, 'w') as out:
        for y, x in zip(Y,X):
            out.write('{0} {1}\n'.format(y, ' '.join(x)))


def convert_dataset_to_sentembeddingformat(dataset, dataset_name):
    tagger = nltk.tag.PerceptronTagger()
    
    X = list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest)
    Y = list(dataset._ytrain) + list(dataset._ydev) + list(dataset._ytest)
    X_tagged = tagger.tag_sents(X)
    X_tagged = [['{0}_{1}'.format(w,t) for w, t in sent] for sent in X_tagged]
    
    # write pos-tagged version
    writeout(X_tagged, Y,'../baselines/sentence_embeddings/preprocess codes/tag_data/{0}.pos'.format(dataset_name))
    # write full version
    writeout(X, Y,'../baselines/sentence_embeddings/preprocess codes/data/{0}/{0}.txt'.format(dataset_name))
    # write test
    writeout(dataset._Xtest, dataset._ytest,'../baselines/sentence_embeddings/preprocess codes/data/{0}/{0}.txttest'.format(dataset_name))
    # write original dev
    writeout(dataset._Xdev, dataset._ydev,'../baselines/sentence_embeddings/preprocess codes/data/{0}/{0}.txtdev'.format(dataset_name))


if __name__ == '__main__':

    semeval_2013 = Semeval_Dataset('../datasets/semeval_2013', None, one_hot=False, rep=words, binary=True)
    semeval_2016 = Semeval_Dataset('../datasets/semeval_2016', None, one_hot=False, rep=words, binary=True)
    books = Amazon_Dataset('../datasets/amazon-multi-domain/books/', None, rep=words, one_hot=False, binary=True)
    dvd = Amazon_Dataset('../datasets/amazon-multi-domain/dvd/', None, rep=words, one_hot=False, binary=True)
    electronics = Amazon_Dataset('../datasets/amazon-multi-domain/electronics/', None, rep=words, one_hot=False, binary=True)
    kitchen = Amazon_Dataset('../datasets/amazon-multi-domain/kitchen_&_housewares/', None, rep=words, one_hot=False, binary=True)
    all_amazon = deepcopy(books)
    all_amazon._Xtrain = list(books._Xtrain) + list(dvd._Xtrain) + list(electronics._Xtrain) + list(kitchen._Xtrain)
    all_amazon._Xde = list(books._Xdev) + list(dvd._Xdev) + list(electronics._Xdev) + list(kitchen._Xdev)
    all_amazon._Xtest = list(books._Xtest) + list(dvd._Xtest) + list(electronics._Xtest) + list(kitchen._Xtest)
    all_amazon._ytrain = list(books._ytrain) + list(dvd._ytrain) + list(electronics._ytrain) + list(kitchen._ytrain)
    all_amazon._ydev = list(books._ydev) + list(dvd._ydev) + list(electronics._ydev) + list(kitchen._ydev)
    all_amazon._ytest = list(books._ytest) + list(dvd._ytest) + list(electronics._ytest) + list(kitchen._ytest)

    

    datasets = [('semeval_2013', semeval_2013), ('semeval_2016', semeval_2016), ('books',books), ('dvd', dvd),
                ('electronics',electronics), ('kitchen', kitchen), ('all-amazon', all_amazon)]

    for name, dataset in datasets:
        print('converting {0} to sentence embedding format...'.format(name))
        convert_dataset_to_sentembeddingformat(dataset, name)
