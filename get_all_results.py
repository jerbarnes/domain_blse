import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def merge(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

if __name__ == '__main__':

    results = {}
    for f in ['no-adaptation.txt', 'mSDA.txt', 'NSCL.txt', 'blse_general.txt']:
        with open('results/'+f) as infile:
            result = json.load(infile)
            results = merge(results, result)

    for tr in results.keys():
        for test in results[tr].keys():
            print('{0} -> {1}'.format(tr, test))
            for model in results[tr][test].keys():
                try:
                    f1 = results[tr][test][model]['f1']
                    print('{0}: {1}'.format(model, f1))
                except KeyError:
                    pass
            print()

    # Check 3 models that have all trains to semevals
    #train = ['books', 'dvd', 'electronics', 'kitchen', 'all']
    #test = ['semeval_2013', 'semeval_2016']
    #models = ['mSDA', 'none','BLSE']
    #colors = ['b','g','r']

    # check 4 models
    train = ['books', 'dvd', 'electronics', 'kitchen']
    test = ['semeval_2013', 'semeval_2016']
    models = ['none', 'mSDA', 'NSCL', 'BLSE']
    model_names = ['NoAd', 'mSDA', 'NSCL', 'BLSE']
    colors = plt.get_cmap('Blues')
    cs = [colors(50), colors(100), colors(150), colors(200)]
    #colors = ['b', 'g', 'y', 'r']

    # check mSDA, none, NSCL for amazons
    #train = ['books', 'dvd', 'electronics', 'kitchen']
    #test = ['books', 'dvd', 'electronics', 'kitchen']
    #models = ['mSDA', 'none', 'NSCL']
    #colors = ['b','g','r']


    for t in test:
        fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
        fig.subplots_adjust(bottom=0.2)
        ax = fig.add_subplot(111)
        data = []

        for tr in train:
            ds = []
            for model in models:
                f1 = results[tr][t][model]['f1']
                ds.append(f1)
            data.append(ds)

        data = np.array(data)

        x = np.arange(len(train))
        i = 0

        font = {'fontname':'Arial', 'size':'10', 'color':'black'}

        # plot each models results
        for j, d in enumerate(data.T):
            ax.bar(x + i, d, width=0.2, color=cs[j], label=model_names[j])
            i += 0.2

        # plot the monodomain versions
        mono2013 = 0.716
        mono2016 = 0.66
        if t == 'semeval_2013':
            ax.plot(np.arange(len(x)+1), [mono2013]*(len(x)+1), label='mono-2013', linestyle='--')
            plt.xticks(x+.4,('B -> S13','D -> S13', 'E -> S13','K -> S13'), **font)
        else:
            ax.plot(np.arange(len(x)+1), [mono2016]*(len(x)+1), label='mono-2016', linestyle='--')
            plt.xticks(x+.4,('B -> S16','D -> S16', 'E -> S16','K -> S16'), **font)

        
        plt.ylim((0.3, 0.899))
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.05),
                    ncol=3, fancybox=True, shadow=True, prop={'size': 12})
        plt.ylabel('Macro F1', **font)
        plt.xlabel('Source -> Target', **font)
        
        plt.show()
        
        
