import sys
sys.path.append('../')
from Utils.Semeval_2013_Dataset import *

def data_to_SCL_format(dataset, outdir, maxn=1000):
    pos = 0
    neg = 0
    X = list(dataset._Xtrain) + list(dataset._Xdev) + list(dataset._Xtest)
    Y = list(dataset._ytrain) + list(dataset._ydev) + list(dataset._ytest)
    with open(outdir + '/semevalUN.txt', 'w') as out:
        out.write('<reviews>')
        for x in X:
            out.write('<review>\n')
            out.write(' '.join(x) + '\n')
            out.write('</review>')
        out.write('</reviews>')
    with open(outdir + '/negative.parsed', 'w') as out:
        for x, y in zip(X, Y):
            if neg < maxn:
                if y == 0:
                    neg += 1
                    out.write('<review>\n')
                    out.write(' '.join(x) + '\n')
                    out.write('</review>')
        out.write('</reviews>')
    with open(outdir + '/positive.parsed', 'w') as out:
        for x, y in zip(X, Y):
            if pos < maxn:
                if y == 2:
                    pos += 1
                    out.write('<review>\n')
                    out.write(' '.join(x) + '\n')
                    out.write('</review>')
        out.write('</reviews>')
