import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import json

if __name__ == '__main__':
    
    divergences = [0.94, 0.94, 0.908, 0.908, 0.873, 0.873, 0.87, 0.87, 0.866, 0.866,
                   0.864, 0.864, 0.814, 0.802, 0.79, 0.775, 0.769, 0.761, 0.748, 0.741]

    domain_pairs = [('dvd', 'books'), ('books', 'dvd'), ('kitchen', 'electronics'), ('electronics', 'kitchen'),
                    ('electronics', 'dvd'), ('dvd', 'electronics'), ('electronics', 'books'), ('books', 'electronics'),
                    ('kitchen', 'dvd'), ('dvd', 'kitchen'), ('kitchen', 'books'), ('books', 'kitchen'), ('dvd', 'semeval_2016'),
                    ('books', 'semeval_2016'), ('dvd', 'semeval_2013'), ('books', 'semeval_2013'), ('electronics', 'semeval_2016'),
                    ('kitchen', 'semeval_2016'), ('electronics', 'semeval_2013'), ('kitchen', 'semeval_2013')]

    blse = [82.2, 81.0, 70.8, 78.3, 76.8, 70.3, 71.3, 71.8, 76.5, 72.3,
            69.0, 73.8, 66.1, 65.2, 67.1, 65.8, 67.0, 62.8, 65.6, 63.9]
    nscl = [77.3, 81.1, 84.0, 84.6, 74.5, 78.1, 71.2, 76.8, 76.3, 80.3,
            73.0, 80.1, 61.9, 61.5, 60.6, 62.8, 60.7, 57.6, 59.2, 50.7]
    msda = [76.1, 78.3, 82.4, 84.5, 71.0, 75.0, 71.9, 74.6, 71.4, 77.4,
            70.0, 78.8, 43.1, 53.1, 45.3, 52.2, 48.2, 55.6, 48.8, 53.2]
    noad = [73.6, 76.0, 81.6, 82.4, 69.2, 70.9, 67.9, 70.0, 70.2, 73.2,
            67.7, 74.0, 63.2, 59.6, 61.5, 61.6, 59.3, 54.2, 60.9, 51.8]


    x = np.arange(len(divergences))
    font = {'fontname':'Arial', 'size':'14', 'color':'black'}
    font2 = {'fontname':'Arial', 'size':'10', 'color':'black'}

    fig = plt.figure(num=None, figsize=(6, 6), dpi=80, facecolor='w', edgecolor='k')
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111)
    ax.plot(x, blse, label='BLSE', linewidth=3)
    ax.plot(x, nscl, label='NSCL', linestyle='--', linewidth=3)
    ax.plot(x, msda, label='mSDA', linestyle='--', linewidth=3)
    ax.plot(x, noad, label='NoAd', linestyle=':', linewidth=3)
    plt.xticks(x, ['d->b (acc)', 'b->d (acc)', 'k->e (acc)', 'e->k (acc)', 'e->d (acc)', 'd->e (acc)', 'e->b (acc)', 'b->e (acc)', 'k->d (acc)', 'd->k (acc)',
                   'k->b (acc)', 'b->k (acc)', 'd->16 (f1)', 'b->16 (f1)', 'd->13 (f1)', 'b->13 (f1)', 'e->16 (f1)','k->16 (f1)', 'e->13 (f1)', 'k->13 (f1)',], rotation=45, **font2)
    plt.yticks(**font)
    plt.ylim((35,99))
    plt.xlabel('Divergence of dataset pairs (similar -> divergent)', **font)
    plt.ylabel('Performance on dataset pairs', **font)
    rcParams.update({'figure.subplot.bottom' : 0.85})
    plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.03),
                    ncol=4, fancybox=True, shadow=True, prop={'size': 14})
    plt.show()
