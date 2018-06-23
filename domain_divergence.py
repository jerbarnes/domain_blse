from Utils.Datasets import *
from Utils.Semeval_2013_Dataset import *

from collections import Counter
import itertools
import codecs
import operator

import scipy.stats

import tabulate

def get_term_dist(docs, vocab, lowercase=True):
    """
    Calculates the term distribution of a list of documents.
    :param docs: a list of tokenized docs; can also contain a single document
    :param vocab: the Vocabulary object
    :param lowercase: lower-case the input data
    :return: the term distribution of the input documents,
             i.e. a numpy array of shape (vocab_size,)
    """
    term_dist = np.zeros(vocab.size)
    for doc in docs:
        for word in doc:
            if lowercase:
                word = word.lower()
            if word in vocab.word2id:
                term_dist[vocab.word2id[word]] += 1

    # normalize absolute freqs to obtain a relative frequency term distribution
    term_dist /= np.sum(term_dist)
    if np.isnan(np.sum(term_dist)):
        # the sum is nan if docs only contains one document and that document
        # has no words in the vocabulary
        term_dist = np.zeros(vocab.size)
    return term_dist


class Vocab:
    """
    The vocabulary class. Stores the word-to-id mapping.
    """
    def __init__(self, max_vocab_size, vocab_path):
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.size = 0
        self.word2id = {}
        self.id2word = {}

    def load(self):
        """
        Loads the vocabulary from the vocabulary path.
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Reading vocabulary from %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_vocab_size:
                    print('Vocab in file is larger than max vocab size. '
                          'Only using top %d words.' % self.max_vocab_size)
                    break
                word, idx = line.split('\t')
                self.word2id[word] = int(idx.strip())
        self.size = len(self.word2id)
        self.id2word = {index: word for word, index in self.word2id.items()}
        assert self.size <= self.max_vocab_size, \
            'Loaded vocab is of size %d., max vocab size is %d.' % (
                self.size, self.max_vocab_size)

    def create(self, texts, lowercase=True):
        """
        Creates the vocabulary and stores it at the vocabulary path.
        :param texts: a list of lists of tokens
        :param lowercase: lowercase the input texts
        """
        assert self.size == 0, 'Vocabulary has already been loaded or built.'
        print('Building the vocabulary...')
        if lowercase:
            print('Lower-casing the input texts...')
            texts = [[word.lower() for word in text] for text in texts]

        word_counts = Counter(itertools.chain(*texts))

        # get the n most common words
        most_common = word_counts.most_common(n=self.max_vocab_size)

        # construct the word to index mapping
        self.word2id = {word: index for index, (word, count)
                        in enumerate(most_common)}
        self.id2word = {index: word for word, index in self.word2id.items()}

        print('Writing vocabulary to %s...' % self.vocab_path)
        with codecs.open(self.vocab_path, 'w', encoding='utf-8') as f:
            for word, index in sorted(self.word2id.items(),
                                      key=operator.itemgetter(1)):
                f.write('%s\t%d\n' % (word, index))
        self.size = len(self.word2id)

def jensen_shannon_divergence(repr1, repr2):
    """Calculates Jensen-Shannon divergence (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)."""
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim):
        # the similarity is -inf if no term in the document is in the vocabulary
        return 0
    return sim

if __name__ == '__main__':

    # Load all the data
    book = Book_Dataset(None, rep=words)
    book = list(book._Xtrain) + list(book._Xdev) + list(book._Xtest)
    book_vocab = Vocab(10000, 'baselines/book-vocab.txt')
    book_vocab.load()

    dvd = DVD_Dataset(None, rep=words)
    dvd = list(dvd._Xtrain) + list(dvd._Xdev) + list(dvd._Xtest)
    dvd_vocab = Vocab(10000, 'baselines/dvd-vocab.txt')
    dvd_vocab.load()

    kitchen = Kitchen_Dataset(None, rep=words)
    kitchen = list(kitchen._Xtrain) + list(kitchen._Xdev) + list(kitchen._Xtest)
    kitchen_vocab = Vocab(10000, 'baselines/kitchen-vocab.txt')
    kitchen_vocab.load()

    electronics = Electronics_Dataset(None, rep=words)
    electronics = list(electronics._Xtrain) + list(electronics._Xdev) + list(electronics._Xtest)    
    electronics_vocab = Vocab(10000, 'baselines/electronics-vocab.txt')
    electronics_vocab.load()

    semeval_2013 = Semeval_Dataset('datasets/semeval_2013', None, rep=words)
    semeval_2013 = list(semeval_2013._Xtrain) + list(semeval_2013._Xdev) + list(semeval_2013._Xtest)
    semeval_2013_vocab = Vocab(10000, 'baselines/semeval2013-vocab.txt')
    semeval_2013_vocab.load()

    semeval_2016 = Semeval_Dataset('datasets/semeval_2016', None, rep=words)
    semeval_2016 = list(semeval_2016._Xtrain) + list(semeval_2016._Xdev) + list(semeval_2016._Xtest)
    semeval_2016_vocab = Vocab(10000, 'baselines/semeval2016-vocab.txt')
    semeval_2016_vocab.load()

    # get the shared vocabulary space
    joint_w2idx = {}
    for v in [book_vocab, dvd_vocab, electronics_vocab,
              kitchen_vocab, semeval_2013_vocab,
              semeval_2016_vocab]:
        for w in v.word2id.keys():
            if w not in joint_w2idx:
                joint_w2idx[w] = len(joint_w2idx)
                
    joint_vocab = Vocab(10000, 'baselines/joint_vocab.txt')
    joint_vocab.size = len(joint_w2idx)
    joint_vocab.word2id = joint_w2idx
    
    # create distributions
    book_dist = get_term_dist(book, joint_vocab)
    dvd_dist = get_term_dist(dvd, joint_vocab)
    electronics_dist = get_term_dist(electronics, joint_vocab)
    kitchen_dist = get_term_dist(kitchen, joint_vocab)
    semeval_2013_dist = get_term_dist(semeval_2013, joint_vocab)
    semeval_2016_dist = get_term_dist(semeval_2016, joint_vocab)
    distributions = [book_dist, dvd_dist, electronics_dist,
                     kitchen_dist, semeval_2013_dist,
                     semeval_2016_dist]

    # calculate distances
    headers=['book', 'dvd', 'electronics',
             'kitchen', 'semeval2013', 'semeval2016']
    distances = [[],[],[],[],[],[]]
    
    for i, source_dist in enumerate(distributions):
        distances[i].append(headers[i])
        for j, target_dist in enumerate(distributions):
            divergence = jensen_shannon_divergence(source_dist, target_dist)
            distances[i].append(divergence)
    
    print(tabulate.tabulate(distances, headers=['']+headers, floatfmt='0.3f'))
    latex = tabulate.tabulate(distances, headers=['']+headers, floatfmt='0.3f',
                              tablefmt='latex')
    with open('./figures/divergence_table.txt', 'w') as out:
        out.write(latex)
