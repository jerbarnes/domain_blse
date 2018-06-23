import argparse
import sys
from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *
import nltk
import numpy as np
from tabulate import tabulate

if __name__ == '__main__':
    
    books = Book_Dataset(None, rep=words, one_hot=False, binary=True)
    dvd = DVD_Dataset(None, rep=words, one_hot=False, binary=True)
    electronics = Electronics_Dataset(None, rep=words, binary=True, one_hot=False)
    kitchen = Kitchen_Dataset(None, rep=words, binary=True, one_hot=False)

    semeval_2013 = Semeval_Dataset('datasets/semeval_2013', None,
                                      binary=True, rep=words,
                                      one_hot=False)

    semeval_2016 = Semeval_Dataset('datasets/semeval_2016', None,
                                      binary=True, rep=words,
                                      one_hot=False)

    books = list(books._Xtrain) + list(books._Xdev) + list(books._Xtest)
    dvd = list(dvd._Xtrain) + list(dvd._Xdev) + list(dvd._Xtest)
    electronics = list(electronics._Xtrain) + list(electronics._Xdev) + list(electronics._Xtest)
    kitchen = list(kitchen._Xtrain) + list(kitchen._Xdev) + list(kitchen._Xtest)
    semeval_2013 = list(semeval_2013._Xtrain) + list(semeval_2013._Xdev) + list(semeval_2013._Xtest)
    semeval_2016 = list(semeval_2016._Xtrain) + list(semeval_2016._Xdev) + list(semeval_2016._Xtest)

    corpora = [('books', books),
               ('dvd', dvd),
               ('electronics', electronics),
               ('kitchen', kitchen),
               ('semeval_2013', semeval_2013),
               ('semeval_2016', semeval_2016),
               ]

    analysis = {}

    for name, corpus in corpora:
        analysis[name] = {}
        
        fd = nltk.FreqDist()
        bigram_fd = nltk.FreqDist()
        num_tokens = 0
        types = set()
        lengths = []
        for ex in corpus:
            fd.update(ex)
            bigrams = nltk.bigrams(ex)
            bigram_fd.update(bigrams)
            num_tokens += len(ex)
            types.update(ex)
            lengths.append(len(ex))
        analysis[name]['num. docs'] = len(lengths)
        analysis[name]['uni-hapaxes'] = len(fd.hapaxes())
        analysis[name]['% uni-hapaxes'] = len(fd.hapaxes()) / len(fd)
        analysis[name]['bi-hapaxes'] = len(bigram_fd.hapaxes())
        analysis[name]['% bi-hapaxes'] = len(bigram_fd.hapaxes()) / len(bigram_fd)
        analysis[name]['TTR'] = len(types) / num_tokens
        analysis[name]['av. length'] = np.mean(lengths)
        analysis[name]['num. tokens'] = num_tokens

    data = []
    for name, corpus in corpora:
        header = list(analysis[name].keys())
        data.append([name] + list(analysis[name].values()))

    print(tabulate(data, headers=header))
