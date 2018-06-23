import argparse
import sys
import os
from subprocess import Popen, STDOUT, PIPE

def collect_lemmas(f):
    sents = []
    sent = []
    for line in f:
        try:
            token, lemma, pos = line.split()
            sent.append(lemma)
        except ValueError:
            sents.append(' '.join(sent))
            sent = []
    return '\n'.join(sents)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="input file")
    parser.add_argument('-o', help="output file")

    args = vars(parser.parse_args())
    infile = args['i']
    outfile = args['o']

    ixa_dir = '/home/jeremy/old/ixa-pipes-1.1.1'

    # preprocess
    proc1 =[l.strip() for l in open(infile)]
    proc2 = []
    for l in proc1:
        if not l.endswith('.'):
            proc2.append(l+' .')
        else:
            proc2.append(l)
            
    proc_out = [l[0].upper()+l[1:] for l in proc2]
    with open('preprocessed.txt', 'w') as out:
        out.write('\n'.join(proc_out))
    
    
    # tokenize
    f = open('preprocessed.txt')
    p = Popen(['java', '-jar', ixa_dir+'/ixa-pipe-tok-1.8.5-exec.jar', 'tok', '-l', 'eu'],
               stdin=f, stderr=STDOUT, stdout=PIPE)
    tok = '\n'.join(p.communicate()[0].decode().splitlines()[3:])
    with open('tok.txt', 'w') as out:
        out.write(tok)

    # lemmatize
    f = open('tok.txt')
    p = Popen(['java', '-jar', ixa_dir + '/ixa-pipe-pos-1.5.1-exec.jar', 'tag', '-m', ixa_dir + '/morph-models-1.5.0/eu/eu-pos-perceptron-epec.bin', '-lm',
               ixa_dir + '/morph-models-1.5.0/eu/eu-lemma-perceptron-epec.bin', '-o', 'conll'],
          stdin=f, stdout=PIPE, stderr=STDOUT)
    processed = p.communicate()[0].decode().splitlines()[3:]
    lemmatized = collect_lemmas(processed)
    with open(outfile, 'w') as out:
        out.write(lemmatized)
    
    
if __name__ == "__main__":

    args = sys.argv
    main(args)
