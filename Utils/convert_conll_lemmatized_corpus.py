import argparse
import sys

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help="input file")
    parser.add_argument('-o', help="output file")

    args = vars(parser.parse_args())
    infile = args['i']
    outfile = args['o']

    f = open(infile)
    
    with open(outfile, 'w') as out:
        sent = []
        for line in f:
            try:
                token, lemma, pos = line.split()
                sent.append(lemma)
            except ValueError:
                out.write(' '.join(sent)+'\n')
                sent = []
    
if __name__ == "__main__":

    args = sys.argv
    main(args)
