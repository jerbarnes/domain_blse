import sys
sys.path.append('../')
import re
import nltk
import numpy as np
import scipy.io as sio
from collections import defaultdict
from twokenize import *

class FEATURE_GENERATOR:

    def __init__(self, stopword_file='../baselines/stopwords.txt'):
        # How many instances should be taken as training data.
        self.load_stop_words(stopword_file)
        pass

    def load_stop_words(self, stopwords_fname):
        """
        Read the list of stop words and store in a dictionary.
        """
        self.stopWords = []
        F = open(stopwords_fname)
        for line in F:
            self.stopWords.append(line.strip())
        F.close()
        pass

    def is_stop_word(self, word):
        """
        If the word is listed as a stop word in self.stopWords,
        then returns True. Otherwise, returns False.
        """
        return (word in self.stopWords)

    def clean(self, line):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`//.:]", " ", line)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.lower()

    def get_tokens(self, line):
        """
        return 
        """
        line = self.clean(line)
        return line.strip().split()
    

    def get_rating_from_label(self, label):
        """
        Set the rating using the label.
        """
        if label == "positive":
            return "positive"
        elif label == "negative":
            return "negative"
        elif label == "unlabeled":
            return None
        pass

    def rem_mentions_urls(self, tokens):
        final = []
        for t in tokens:
            if t.startswith('@'):
                final.append('@')
            elif t.startswith('http'):
                final.append('url')
            else:
                final.append(t)
        return final

    def process_twitter_file(self, fname, labels=False):
        # if labels == True, only return the feature vectors
        # of the positive and negative examples
        count = 0
        feature_vectors = []
        with open(fname) as F:
            for line in F:
                try:
                    idx, idx2, label, tweet = line.split('\t')
                except ValueError:
                    idx, label, tweet = line.split('\t')
                if 'positive' in label:
                    label = 1
                elif 'negative' in label:
                    label = 0
                else:
                    if labels:
                        continue
                    else:
                        pass
                tweet = self.clean(tweet)
                tokens = self.rem_mentions_urls(tokenize(tweet))
                fv = self.get_features(tokens, rating=None)
                feature_vectors.append((label, fv))
                count += 1
        print(fname, len(feature_vectors))
        return feature_vectors
                
                

    def process_amazon_file(self, fname, label=None, encoding='utf8'):
        """
        Open the file fname, generate all the features and return
        as a list of feature vectors.
        """
        feature_vectors = [] #List of feature vectors.
        F = open(fname, encoding=encoding)
        line = F.readline()
        inReview = False
        count = 0
        tokens = []
        while line:
            if line.startswith('^^ <?xml version="1.0"?>'):
                line = F.readline()
                continue
            if line.startswith("<review>"):
                inReview = True
                tokens = []
                line = F.readline()
                continue
            if inReview and line.startswith("<rating>"):
                # Do not uncomment the following line even if you are not
                # using get_rating_from_score because we must skip the rating line.
                ratingStr = F.readline()
                line = F.readline() #skipping the </rating>
                continue
            if inReview and line.startswith("<review_text>"):
                while line:
                    if line.startswith("</review_text>"):
                        break
                    if len(line) > 1 and not line.startswith("<review_text>"):
                        curTokens = self.get_tokens(line.strip())
                        if curTokens:
                            tokens.extend(curTokens)
                    line = F.readline()                    
            if inReview and line.startswith("</review>"):
                inReview = False
                # generate feature vector from tokens.
                # Do not use rating related features to avoid overfitting.
                fv = self.get_features(tokens, rating=None)
                feature_vectors.append((label, fv))
                tokens = []
                count += 1
            line = F.readline()
        # write the final lines if we have not seen </review> at the end.
        if inReview:
            count += 1
        F.close()
        print(fname, len(feature_vectors))
        return feature_vectors
    

    def get_features(self, tokens, rating=None):
        """
        Create a feature vector from the tokens and return it.
        """
        fv = defaultdict(int)       
        # generate unigram features
        for token in tokens:
            if not self.is_stop_word(token):
                fv[token] += 1
                
        # generate bigram features.
        for i in range(len(tokens) - 1):
            bigram = "%s__%s" % (tokens[i], tokens[i+1])
            if not self.is_stop_word(tokens[i]) and not self.is_stop_word(tokens[i+1]):
                fv[bigram] += 1
        return fv


def get_most_common_features(num_features=30000,
                             files=['../datasets/sorted_data/books/positive.review',
                                    '../datasets/sorted_data/books/negative.review',
                                    '../datasets/sorted_data/dvd/positive.review',
                                    '../datasets/sorted_data/dvd/negative.review',
                                    '../datasets/sorted_data/electronics/positive.review',
                                    '../datasets/sorted_data/electronics/negative.review',
                                    '../datasets/sorted_data/kitchen_&_housewares/positive.review',
                                    '../datasets/sorted_data/kitchen_&_housewares/negative.review',
                                    '../datasets/semeval_2013/train.tsv',
                                    '../datasets/semeval_2013/dev.tsv',
                                    '../datasets/semeval_2013/test.tsv',
                                    '../datasets/semeval_2016/train.tsv',
                                    '../datasets/semeval_2016/dev.tsv',
                                    '../datasets/semeval_2016/test.tsv']):
    
    feature_generator = FEATURE_GENERATOR()
    freq_dist = nltk.FreqDist()

    feature_sets = []
    
    for file in files:
        if 'semeval' in file:
            feature_vecs = feature_generator.process_twitter_file(file)
            feature_sets.extend(feature_vecs)
            for label, fv in feature_vecs:
                freq_dist.update(fv)
        else:
            if 'positive' in file: 
                feature_vecs = feature_generator.process_amazon_file(file, 1, encoding='latin')
            else:
                feature_vecs = feature_generator.process_amazon_file(file, 0, encoding='latin')
            feature_sets.extend(feature_vecs)
            for label, fv in feature_vecs:
                freq_dist.update(fv)
    return feature_sets, freq_dist.most_common(num_features)

def create_features2idx(features):
    f2idx = {}
    for f, count in features:
        f2idx[f] = len(f2idx)
    return f2idx

def bag_of_features(features, f2idx):
    vec = np.zeros(len(f2idx))
    for f, count in features.items():
        if f in f2idx:
            vec[f2idx[f]] = count
    return vec

if __name__ == '__main__':

    # Parameters
    N = 30000

    # Find most common N features across and create a feature2index dictionary
    print('getting most common {0} features...'.format(N))
    feature_sets, features = get_most_common_features(N)
    f2idx = create_features2idx(features)

    fg = FEATURE_GENERATOR()
    labels, feature_x = zip(*feature_sets)
    #xx = np.array([bag_of_features(x, f2idx) for x in feature_x])
    print()


    # Convert amazon files to bag-of-features representation
    print('converting amazon...')
    amazon = []

    for domain in ['books', 'dvd', 'electronics', 'kitchen_&_housewares']:
        amazon.extend(fg.process_amazon_file('../datasets/sorted_data/{0}/positive.review'.format(domain), 1, encoding='latin'))
        amazon.extend(fg.process_amazon_file('../datasets/sorted_data/{0}/negative.review'.format(domain), 0, encoding='latin'))
    
    yy, xx = zip(*amazon)

    xx = np.array([bag_of_features(i, f2idx) for i in xx])
    xx = xx.transpose()
    yy = np.array(yy)
    yy = yy.reshape(yy.shape[0], 1)
    print()

    # Convert semeval dataset to bag-of-features representation
    print('converting semeval_2013...')
    semeval_2013 = []
    # We only include the test sets, as these are the ones we test on for the other experiments
    for file in ['../datasets/semeval_2013/test.tsv', '../datasets/semeval_2013/dev.tsv', '../datasets/semeval_2013/train.tsv']:
        semeval_2013.extend(fg.process_twitter_file(file, labels=True))

    trg_y, trg_x = zip(*semeval_2013)

    trg_x = np.array([bag_of_features(i, f2idx) for i in trg_x])
    trg_x = trg_x.transpose()
    trg_y = np.array(trg_y)
    trg_y = trg_y.reshape(trg_y.shape[0], 1)
    print()

    # Convert semeval dataset to bag-of-features representation
    print('converting semeval_2016...')
    semeval_2016 = []
    # We only include the test sets, as these are the ones we test on for the other experiments
    for file in ['../datasets/semeval_2016/test.tsv', '../datasets/semeval_2016/dev.tsv', '../datasets/semeval_2016/train.tsv']:
        semeval_2016.extend(fg.process_twitter_file(file, labels=True))

    trg_y2, trg_x2 = zip(*semeval_2016)

    trg_x2 = np.array([bag_of_features(i, f2idx) for i in trg_x2])
    trg_x2 = trg_x2.transpose()
    trg_y2 = np.array(trg_y2)
    trg_y2 = trg_y2.reshape(trg_y2.shape[0], 1)
    print()

    offset = np.array([0, 2000,4000,6000,8000, 8000+len(semeval_2013), 8000+len(semeval_2013)+len(semeval_2016) ])

    # Save all representations for use in mSDA experiment
    print('saving data to../datasets/amazon+semevalfull.mat')
    sio.savemat('../datasets/amazon+semevalfull.mat', mdict={'xx':xx, 'yy':yy,
                                                         'trg_x':trg_x, 'trg_y':trg_y,
                                                         'trg_x2':trg_x2, 'trg_y2':trg_y2,
                                                         'offset':offset})