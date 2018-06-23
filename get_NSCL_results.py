import os
import json
import re

def get_best_f1(file):
    best_f1 = 0
    best_c = 0
    best_pivot = 0
    best_dim = 0
    for line in open(file):
        try:
            f1 = float(re.findall('macro f1 = (0.[0-9]*) ', line)[0])
            if f1 > best_f1:
                best_f1 = f1
                best_c = float(re.findall('c_parm = (0.[0-9]*)', line)[0])
                best_pivot = int(re.findall('pivot num = ([0-9]*)', line)[0])
                best_dim = int(re.findall('dim = ([0-9]*)', line)[0])
        except IndexError:
            pass
    return best_f1, best_c, best_pivot, best_dim

if __name__ == '__main__':

    results = {}

    for train in ['books', 'dvd', 'electronics', 'kitchen', 'semeval_2013', 'semeval_2016']:
        results[train] = {}
        for test in ['books', 'dvd', 'electronics', 'kitchen', 'semeval_2013', 'semeval_2016']:
            if train != test:
                results[train][test] = {}
                results[train][test]['NSCL'] = {}

                try:
                    infile = os.path.join('baselines', 'NSCL', train + '_to_' + test, 'results', train + '_to_' + test)
                    best_f1, best_c, best_pivot, best_dim = get_best_f1(infile)
                    results[train][test]['NSCL']['f1'] = best_f1
                    results[train][test]['NSCL']['c'] = best_c
                    results[train][test]['NSCL']['pivot'] = best_pivot
                    results[train][test]['NSCL']['dim'] = best_dim
                except FileNotFoundError:
                    pass

    with open('results/NSCL.txt', 'w') as out:
        json.dump(results, out)
            
