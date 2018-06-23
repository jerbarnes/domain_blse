import json
import os
import matplotlib.pyplot as plt
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
    for f in os.listdir('results'):
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


    # check 4 models (currently NSCL doesn't have all to semevals)
    train = ['books', 'dvd', 'electronics', 'kitchen']
    test = ['books', 'dvd', 'electronics', 'kitchen', 'semeval_2013', 'semeval_2016']
    models = ['none', 'mSDA', 'NSCL', 'BLSE']
    colors = ['b', 'g', 'y', 'r']


    for t in test:
        data = []
        trains = []
        for tr in train:
            if t != tr:
                trains.append(tr)
                ds = []
                for model in models:
                    f1 = results[tr][t][model]['f1']
                    ds.append(f1)
                data.append(ds)

        data = np.array(data)

        x = np.arange(len(trains))
        i = 0

        # plot each models results
        for j, d in enumerate(data.T):
            plt.bar(x + i, d, width=0.2, color=colors[j], label=models[j])
            i += 0.2

        # plot the monodomain versions
        mono = results[t][t]['none']['f1']
        plt.plot(np.arange(len(x)+1), [mono]*(len(x)+1), label='mono', linestyle='-')

        plt.ylim((0.3, 0.99))
        plt.xticks(x+.4, (trains))
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.02),
                    ncol=5, fancybox=True, shadow=True)
        plt.title('Test: {0}'.format(t))
        plt.ylabel('Macro F1')
        plt.xlabel('Source Domain')

        plt.show()


