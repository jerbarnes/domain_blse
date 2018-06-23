import sys
import argparse
from sklearn.manifold import TSNE
import numpy as np
from WordVecs import *
import matplotlib.pyplot as plt

def plot_categories(infile, src_vecs, show_labels=False, outfile=None):
    tsne = TSNE(perplexity=40, init='pca', n_iter=5000)
    fig = plt.figure(figsize=(18,18))
    if outfile:
        pass
    else:
        figlegend = plt.figure()
    ax = fig.add_subplot(111)
    words = {}
    for line in open(infile):
        s = line.split()
        # first word is the name of the category
        words[s[0]] = s[3:]

    # get only the words that appear in the vectors
    words_in_vecs = {}
    for category in words.keys():
        ws = []
        for w in words[category]:
            try:
                _ = src_vecs[w]
                ws.append(w)
            except:
                pass
        words_in_vecs[category] = ws


    colors = {'pos':'b', 'neg':'r',
              'animals':'g', 'grammatical':'y',
              'verbs':'orange', 'transport':'black'}

    wordcategories = {}
    for cat, words in words_in_vecs.items():
        for w in words:
            if cat == 'pos1' or cat == 'pos2':
                wordcategories[w] = 'pos'
            else:
                wordcategories[w] = cat
    
    labels = []
    vectors = []

    # set up the labels and vectors
    for cat in words_in_vecs.values():
        for w in cat:
            labels.append(w)
            vectors.append(src_vecs[w])

    
    # perform tsne to reduce the size
    new_values = tsne.fit_transform(vectors)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])    

    legend_categories = []
    plotted_categories = []
    for i in range(len(x)):
        label = labels[i]
        cat = wordcategories[label]
        color = colors[cat]
        if cat not in legend_categories:
            pcat = ax.scatter(x[i],y[i], c=color, label=cat)
            plotted_categories.append(pcat)
            legend_categories.append(cat)
        else:
            ax.scatter(x[i],y[i], c=color)
        if show_labels:
            ax.annotate(label,
                     xy=(x[i],y[i]),
                     xytext=(5,2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        figlegend.legend(plotted_categories, legend_categories, fontsize=40)
        plt.show()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    parser = argparse.ArgumentParser(description='plot tsne view of vectors')
    parser.add_argument('-e', '--embeddings', help='vector file')
    parser.add_argument('-c', '--categories', default=None)
    parser.add_argument('-l', '--show_labels', default=False, type=str2bool)
    args = parser.parse_args()    

    vecs = WordVecs(args.embeddings)
    plot_categories(args.categories, vecs, show_labels=args.show_labels)
    
if __name__ == '__main__':
    main()
