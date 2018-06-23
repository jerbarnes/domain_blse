import tr
import sentiment
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='domain adaptation from "books, kitchen, dvd, electronics, semeval_2013, semeval_2016"')
    parser.add_argument('-tr', help="training domain (default =  books)", default='books')
    parser.add_argument('-te', help="test_domain (default =  kitchen)", default='kitchen')
    parser.add_argument('-dim', help='number of hidden units (default = 100)', default=100, type=int)
    parser.add_argument('-min', help='minimum frequency for pivots (default = 10)', default=10, type=int)
    parser.add_argument('-piv', help='number of pivots (default = 500)', default=500, type=int)
    parser.add_argument('-c', help='C parameter for svm (default = 0.1)', default=0.1, type=float)
    args = parser.parse_args()



    print('Domain adaptation from {0} to {1}'.format(args.tr, args.te))

    # making a shared representation for both source domain and target domain
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth parameter: the embedding dimension, identical to the hidden layer dimension

    tr.train(args.tr, args.te, args.dim, args.min, args.piv)

    # learning the classifier in the source domain and testing in the target domain
    # the results, weights and all the meta-data will appear in source-target directory
    # first param: the source domain
    # second param: the target domain
    # third param: number of pivots
    # fourth param: appearance threshold for pivots in source and target domain
    # fifth param: the embedding dimension identical to the hidden layer dimension
    # sixth param: we use logistic regression as our classifier, it takes the const C for its learning

    sentiment.sent(args.tr, args.te, args.dim, args.min, args.piv, args.c)

