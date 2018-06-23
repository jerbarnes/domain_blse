import codecs
import numpy as np
from nltk.corpus import stopwords
import csv
import gensim
import logging
import Cython
import os
from gensim import models
from gensim.models import word2vec
import re
import xml.etree.ElementTree as ET
import pickle


punctuations = [")", "(", "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--",
                "...", ";"]
stops = stopwords.words('english')
encoding_ = 'utf-8'

def getClear_full(sentence):
    r = re.findall(r'\b\w+\b', sentence.lower())
    r = " ".join(r)
    return r

def getClear(sentence,bigram):
    r = re.findall(r'\b\w+\b', sentence.lower())
    length=len(r)

    i=0
    while(i<length-1):
        if (r[i]+'_'+r[i+1]) in bigram:
            r.insert(i+1,r[i]+'_'+r[i+1])
           # r[i]=r[i]+'_'+r[i+1]
           # del r[i+1]
            length=length+1


        i=i+1
    r = " ".join(r)
    return r

def getStopWords(src, dest,pivot_num,pivot_min_st):
    filename = src + "_to_" + dest + "/" + "pivot_names/pivot_names_" + src + "_" + dest + "_" + str(
        pivot_num) + "_" + str(pivot_min_st)
    with open((filename), 'rb') as f:
        my_list = pickle.load(f)

    unigram = []
    bigram = []
    cap_unigrams = []
    cap_bigrams = []
    count = 0
    for feature in my_list:

        if (len(feature.split()) == 1):
            unigram.append(feature)
        if (len(feature.split()) == 2):
            word=feature.split()
            temp=word[0]+'_'+word[1]
            bigram.append(temp)
    return unigram,bigram

def XML2arrayRAW(path):
    reviews = []
    negReviews = []
    posReviews = []
    tree = ET.parse(path)
    root = tree.getroot()
    for rev in root.iter('review'):
        reviews.append(rev.text)


    return reviews

def XML2arrayRAW_org(neg_path, pos_path):
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

def makeFileForWord2Vec(src, dest,pivot_num,pivot_min_st):
    unigram, bigram = getStopWords(src, dest,pivot_num,pivot_min_st)
    reviews_labels_neg = XML2arrayRAW("../data/"+src+"/negative.parsed")
    reviews_labels_pos = XML2arrayRAW("../data/" + src + "/positive.parsed")
    reviews_unlabeld_src= XML2arrayRAW("../data/"+src+"/"+src+"UN.txt")
    reviews_unlabeld_dest = XML2arrayRAW("../data/"+dest+"/"+dest+"UN.txt")
    all= reviews_labels_neg+reviews_labels_pos+reviews_unlabeld_src+reviews_unlabeld_dest
    filename = src + "_to_" + dest + "/" + "W2V/w2v_" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    f = open(filename, 'w')
    for r in all:
        t = getClear(r,bigram)
        f.write("%s\n" % t)

    f.close()  # you can omit in most cases as the destructor will call it

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        i=0
        for line in codecs.open(self.fname, 'r', encoding=encoding_):
            line = line.strip()
            linelist = []
            #print i
            #i=i+1
            lineL = line.split(' ')
            for w in lineL:

                w = w.lower()
                if w.isdigit():
                    w = '@'
                linelist.append(w)
            yield linelist




def wo2ve(src,dest,pivot_num,pivot_min_st,dim):
   # makeFileForWord2Vec("../data/dvd/dvdUN.txt","un_dvd")
  #  makeFileForWord2Vec("../data/electronics/electronicsUN.txt","un_electronics")
  #  makeFileForWord2Vec("../data/kitchen/kitchenUN.txt","un_kitchen")
   # makeFileForWord2Vec("../data/books/booksUN.txt","un_books")
    makeFileForWord2Vec(src,dest,pivot_num,pivot_min_st)


    filename = src + "_to_" + dest + "/" + "W2V/w2v_" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(
        pivot_min_st)

    sentences = MySentences(filename)  # a memory-friendly iterator


    new_model = gensim.models.Word2Vec(sentences,min_count=10,size=dim,workers=8)



    filename = src + "_to_" + dest + "/" + "pivot_names/pivot_names_" + src + "_" + dest + "_" + str(pivot_num) + "_" + str(pivot_min_st)

    with open(filename, 'rb') as f:
        my_list = pickle.load(f)

    pivot_mat = np.zeros((pivot_num, dim))
    i=0
    for p in my_list:



        if p.isdigit():
            pivot_mat[i]=new_model['@']
            i=i+1
            continue


        if (len(p.split()) == 1):
            pivot_mat[i]=new_model[p]
        if (len(p.split()) == 2):
            temp=p.split()
            feature = temp[0]+'_'+temp[1]
            pivot_mat[i] = new_model[feature]
        i=i+1



    filename = src + "_to_" + dest + "/" +"pivot_mat/"+"pivot_mat_"+src+"_"+dest+"_"+str(pivot_num)+"_"+str(pivot_min_st)+"_"+str(dim)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(pivot_mat, f)


'''
   quick_new_model = gensim.models.Word2Vec.load('_quick_mymodel')
   print quick_new_model.similarity('recommended', 'suggest')
   print quick_new_model.similarity('woman', 'horse')
   print new_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
   print quick_new_model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)
   model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
   print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10)

   #model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
   '''
