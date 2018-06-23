import argparse
import sys
from copy import deepcopy
sys.path.append('../joint_bilingual_sentiment_embeddings/')
from BilingualNetwork import *
from sklearn.metrics import f1_score
from Utils.Semeval_2013_Dataset import *

def keep_n_models(weightdir, n=10):
    """
    Deletes all but n-best model weights in a directory
    : param n: number of models to keep
    """
    keep = [(0,'')]
    remove = []
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        batch = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        alp = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
        acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if acc > keep[-1][0]:
            keep.append((acc, file))
            keep = sorted(keep,reverse=True)
            if len(keep) > n:
                candidate = keep.pop()
                if candidate != (0, ''):
                    remove.append(candidate)
        else:
            remove.append((acc, file))
    #print(len(remove))
    for acc, file in remove:
        try:
            os.remove(os.path.join(weightdir, file))
        except OSError:
            pass
            print("can't find {0}".format(os.path.join(weightdir, file)))
    print('Kept {0} files'.format(n))
    for acc, file in keep:
        print('{0} {1}'.format(acc, file))

def to_array(X, n=2):
    """
    Converts a list scalars to an array of size len(X) x n
    >>> to_array([0,1], n=2)
    >>> array([[ 1.,  0.],
               [ 0.,  1.]])
    """
    return np.array([np.eye(n)[x] for x in X])

def per_class_f1(y, pred):
    """
    Returns the per class f1 score.
    Todo: make this cleaner.
    """
    
    num_classes = len(set(y))
    y = to_array(y, num_classes)
    pred = to_array(pred, num_classes)
    
    results = []
    for j in range(num_classes):
        class_y = y[:, j]
        class_pred = pred[:, j]
        f1 = f1_score(class_y, class_pred, average='binary')
        results.append([f1])
    return np.array(results)


class BLSE_domain(nn.Module):
    
    def __init__(self, src_vecs, trg_vecs, pdataset,
                 src_dataset, trg_dataset,
                 output_dim=5):
        super(BLSE_domain, self).__init__()
        
        # Embedding matrices
        self.semb = nn.Embedding(src_vecs.vocab_length, src_vecs.vector_size)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs._matrix))
        self.sw2idx = src_vecs._w2idx
        self.sidx2w = src_vecs._idx2w
        self.temb = nn.Embedding(trg_vecs.vocab_length, trg_vecs.vector_size)
        self.temb.weight.data.copy_(torch.from_numpy(trg_vecs._matrix))
        self.tw2idx = trg_vecs._w2idx
        self.tidx2w = trg_vecs._idx2w
        # Projection vectors
        self.m = nn.Linear(src_vecs.vector_size, src_vecs.vector_size, bias=False)
        self.mp = nn.Linear(trg_vecs.vector_size, trg_vecs.vector_size, bias=False)
        # Classifier
        self.clf2 = nn.Linear(src_vecs.vector_size, src_vecs.vector_size)
        self.clf = nn.Linear(src_vecs.vector_size, output_dim)
        # Loss Functions
        self.criterion = nn.CrossEntropyLoss()
        self.proj_criterion = nn.CosineSimilarity()
        # Optimizer
        self.optim = torch.optim.Adam(self.parameters() )
        # Datasets
        self.pdataset = pdataset
        self.src_dataset = src_dataset
        self.trg_dataset = trg_dataset

        # History
        self.history = {'loss':  [], 'dev_cosine': [], 'dev_f1': [], 'cross_f1': []}
        self.semb.weight.requires_grad = False
        self.temb.weight.requires_grad = False

    def dump_weights(self, outfile):
        w1 = self.m.weight.data.numpy()
        w2 = self.mp.weight.data.numpy()
        w3 = self.clf.weight.data.numpy()
        w4 = self.clf2.weight.data.numpy()
        b = self.clf.bias.data.numpy()
        b2 = self.clf2.bias.data.numpy()
        np.savez(outfile, w1, w2, w3, w4, b, b2)

    def load_weights(self, weight_file):
        f = np.load(weight_file)
        self.m.weight.data.copy_(torch.from_numpy(f['arr_0']))
        self.mp.weight.data.copy_(torch.from_numpy(f['arr_1']))
        self.clf.weight.data.copy_(torch.from_numpy(f['arr_2']))
        self.clf2.weight.data.copy_(torch.from_numpy(f['arr_3']))
        self.clf.bias.data.copy_(torch.from_numpy(f['arr_4']))
        self.clf2.bias.data.copy_(torch.from_numpy(f['arr_5']))

    def project(self, x, y):
        """
        Project into shared space.
        """
        x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
        y_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in y]))
        x_embedd = self.semb(Variable(x_lookup))
        y_embedd = self.temb(Variable(y_lookup))
        x_proj = self.m(x_embedd)
        y_proj = self.mp(y_embedd)
        return x_proj, y_proj

    def project_one(self, x, src=True):
        if src:
            x_lookup = torch.LongTensor(np.array([self.sw2idx[w] for w in x]))
            x_embedd = self.semb(Variable(x_lookup))
            x_proj = self.m(x_embedd)
        else:
            x_lookup = torch.LongTensor(np.array([self.tw2idx[w] for w in x]))
            x_embedd = self.temb(Variable(x_lookup))
            x_proj = self.mp(x_embedd)
        return x_proj
    
    def projection_loss(self, x, y):
        x_proj, y_proj = self.project(x, y)
        # CCA
        #loss = pytorch_cca_loss(x_proj, y_proj)

        # Mean Squared Error
        loss = mse_loss(x_proj, y_proj)

        # Cosine Distance
        #loss = (1 - self.proj_criterion(x_proj, y_proj)).mean()
        return loss

    def idx_vecs(self, sentence, model):
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except:
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        return [self.idx_vecs(s, model) for s in X]

    def ave_vecs(self, X, src=True):
        vecs = []
        if src:
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            idxs = self.lookup(X, self.tw2idx)
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def predict(self, x, src=True):
        x = self.ave_vecs(x, src)
        if src:
            x_proj = self.m(x)
        else:
            x_proj = self.mp(x)
        x_proj = F.relu(self.clf2(x_proj))
        out = F.softmax(self.clf(x_proj))
        return out

    def classification_loss(self, x, y, src=True):
        pred = self.predict(x, src=src)
        y = Variable(torch.from_numpy(y))
        loss = self.criterion(pred, y)
        return loss

    def full_loss(self, proj_x, proj_y, class_x, class_y,
                  alpha=.5):
        """
        This is the combined projection and classification loss
        alpha controls the amount of weight given to each
        loss term.
        """
    
        proj_loss = self.projection_loss(proj_x, proj_y)
        class_loss = self.classification_loss(class_x, class_y, src=True)
        return alpha * proj_loss + (1 - alpha) * class_loss

    def fit(self, proj_X, proj_Y,
            class_X, class_Y,
            weight_dir='models',
            batch_size=40,
            epochs=100,
            alpha=0.5):
        num_batches = int(len(class_X) / batch_size)
        best_f1 = 0
        num_epochs = 0
        for i in range(epochs):
            idx = 0
            num_epochs += 1
            for j in range(num_batches):
                cx = class_X[idx:idx+batch_size]
                cy = class_Y[idx:idx+batch_size]
                idx += batch_size
                self.optim.zero_grad()
                clf_loss = self.classification_loss(cx, cy)
                proj_loss = self.projection_loss(proj_X, proj_Y)
                loss = alpha * proj_loss + (1 - alpha) * clf_loss
                loss.backward()
                self.optim.step()
            if i % 1 == 0:
                # check cosine distance between dev translation pairs
                xdev = self.pdataset._Xdev
                ydev = self.pdataset._ydev
                xp, yp = self.project(xdev, ydev)
                score = cos(xp, yp)

                # check source dev f1
                xdev = self.src_dataset._Xdev
                ydev = self.src_dataset._ydev
                xp = self.predict(xdev).data.numpy().argmax(1)
                # macro f1
                dev_f1 = per_class_f1(ydev, xp).mean()

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    weight_file = os.path.join(weight_dir, '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}crossf1'.format(num_epochs, batch_size, alpha, best_f1))
                    self.dump_weights(weight_file)


                sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}'.format(
                    i, loss.data[0], score.data[0], dev_f1))
                sys.stdout.flush()
                self.history['loss'].append(loss.data[0])
                self.history['dev_cosine'].append(score.data[0])
                self.history['dev_f1'].append(dev_f1)

    def get_most_probable_translations(self, src_word, n=5):
        px = self.m(self.semb.weight)
        py = self.mp(self.temb.weight)
        preds = torch.mm(py, (px[self.sw2idx[src_word]]).unsqueeze(1))
        preds = preds.squeeze(1)
        preds = preds.data.numpy()
        return [self.tidx2w[i] for i in preds.argsort()[-n:]]

    def plot(self, title=None, outfile=None):
        h = self.history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h['dev_cosine'], label='translation_cosine')
        ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
        ax.plot(h['cross_f1'], label='target_f1', linestyle=':')
        ax.set_ylim(-.5, 1.4)
        ax.legend(
                loc='upper center', bbox_to_anchor=(.5, 1.05),
                ncol=3, fancybox=True, shadow=True)
        if title:
            ax.title(title)
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()
    

    def confusion_matrix(self, X, Y, src=True):
        pred = self.predict(X, src=src).data.numpy().argmax(1)
        cm = confusion_matrix(Y, pred, sorted(set(Y)))
        print(cm)

    def evaluate(self, X, Y, src=True, outfile=None):
        pred = self.predict(X, src=src).data.numpy().argmax(1)
        acc = accuracy_score(Y, pred)
        f1 = per_class_f1(Y, pred).mean()
        if outfile:
            with open(outfile, 'w') as out:
                for i in pred:
                    out.write('{0}\n'.format(i))
        else:
            return acc, f1


def get_best_run(weightdir, batch_size=None, alpha=None):
    """
    This returns the best dev acc, parameters, and weights from the models
    found in the weightdir.
    """
    best_params = []
    best_acc = 0.0
    best_weights = ''
    for file in os.listdir(weightdir):
        epochs = int(re.findall('[0-9]+', file.split('-')[-4])[0])
        batch = int(re.findall('[0-9]+', file.split('-')[-3])[0])
        alp = float(re.findall('0.[0-9]+', file.split('-')[-2])[0])
        acc = float(re.findall('0.[0-9]+', file.split('-')[-1])[0])
        if batch_size and alpha:
            if batch == batch_size and alp == alpha:
                if acc > best_acc:
                    best_params = [epochs, batch, alp]
                    best_acc = acc
                    weights = os.path.join(weightdir, file)
                    best_weights = weights
        elif batch_size:
            if batch == batch_size:
                if acc > best_acc:
                    best_params = [epochs, batch, alp]
                    best_acc = acc
                    weights = os.path.join(weightdir, file)
                    best_weights = weights
        elif alpha:
            if alp == alpha:
                if acc > best_acc:
                    best_params = [epochs, batch, alp]
                    best_acc = acc
                    weights = os.path.join(weightdir, file)
                    best_weights = weights
        else:
            if acc > best_acc:
                best_params = [epochs, batch, alp]
                best_acc = acc
                weights = os.path.join(weightdir, file)
                best_weights = weights
                
    return best_acc, best_params, best_weights


def print_gold(gold_y, outfile):
    with open(outfile, 'w') as out:
        for l in gold_y:
            out.write('{0}\n'.format(l))


def print_info(train, test, alpha, batch_size):
    print('{0} --> {1}'.format(train, test))
    print('batch: {0}'.format(batch_size))
    print('alpha: {0}'.format(alpha))

def main():
    parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-outdim', help="number of classes to predict (default: 2)", default=2, type=int)
    parser.add_argument('-epochs', default=200, type=int, help="training epochs (default: 200)")
    parser.add_argument('-trans', help='translation pairs (default: Bing Liu Sentiment Lexicon Translations)', default='lexicons/general_vocab.txt')
    parser.add_argument('-p', '--predictions_dir', help="directory to write predictions", default='predictions/')
    parser.add_argument('-savedir', default='models/amazon-vecs/', help="where to dump weights during training (default: ./models)")
    args = parser.parse_args()

    books = Book_Dataset(None, rep=words, one_hot=False, binary=True)
    dvd = DVD_Dataset(None, rep=words, one_hot=False, binary=True)
    electronics = Electronics_Dataset(None, rep=words, binary=True, one_hot=False)
    kitchen = Kitchen_Dataset(None, rep=words, binary=True, one_hot=False)

    Xtrain = [list(d._Xtrain) for d in [books, dvd, electronics, kitchen]]
    Xtrain = [w for l in Xtrain for w in l]
    Xdev = [list(d._Xdev) for d in [books, dvd, electronics, kitchen]]
    Xdev = [w for l in Xdev for w in l]

    ytrain = [list(d._ytrain) for d in [books, dvd, electronics, kitchen]]
    ytrain = np.array([w for l in ytrain for w in l])
    ydev = [list(d._ydev) for d in [books, dvd, electronics, kitchen]]
    ydev = np.array([w for l in ydev for w in l])

    train_dataset = deepcopy(books)
    train_dataset._Xtrain = Xtrain
    train_dataset._Xdev = Xdev
    train_dataset._ytrain = ytrain
    train_dataset._ydev = ydev

    semeval_2013 = Semeval_Dataset('datasets/semeval_2013', None,
                                      binary=True, rep=words,
                                      one_hot=False)

    semeval_2016 = Semeval_Dataset('datasets/semeval_2016', None,
                                      binary=True, rep=words,
                                      one_hot=False)


    data =  [('books', books, 'embeddings/amazon-sg-300.txt'),
             ('semeval_2016', semeval_2016, 'embeddings/twitter_embeddings.txt'),
             ('semeval_2013', semeval_2013, 'embeddings/twitter_embeddings.txt'),
             ('dvd', dvd, 'embeddings/amazon-sg-300.txt'),
             ('electronics', electronics, 'embeddings/amazon-sg-300.txt'),
             ('kitchen', kitchen, 'embeddings/amazon-sg-300.txt')
             #('all', train_dataset, 'embeddings/amazon-sg-300.txt'),
             ]

    # iterate over all datasets as train and test
    for tr_name, train, source_embedding_file in data:
        for test_name, test, target_embedding_file in data:
            if tr_name != test_name:

                # If we use the ppmi lexicons, each source-target combination has its own lexicon
                # Otherwise, we use the same lexicon for all
                if 'ppmi' in args.trans:
                    trans = 'lexicons/{0}_to_{1}_ppmi_lexicon.txt'.format(tr_name, test_name)
                    print('Using {0} as projection lexicon'.format(trans))
                else:
                    trans = args.trans
                    print('Using {0} as projection lexicon'.format(trans))

                # open source and target embeddings
                src_vecs = WordVecs(source_embedding_file)
                trg_vecs = WordVecs(target_embedding_file)
                pdataset = ProjectionDataset(trans, src_vecs, trg_vecs)

                savedir = os.path.join(args.savedir,'{0}/{1}'.format(tr_name, test_name))

                for alpha in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    for batch_size in [100, 200, 500]:

                        print_info(tr_name, test_name, alpha, batch_size)

                        clf = BLSE_domain(src_vecs, trg_vecs,  pdataset,
                              train, test, output_dim=2)

                        clf.fit(pdataset._Xtrain, pdataset._ytrain,
                            train._Xtrain, train._ytrain,
                            weight_dir=savedir,
                            batch_size=batch_size,
                            epochs=args.epochs,
                            alpha=alpha)

                        best_acc, best_params, best_weights = get_best_run(savedir, alpha=alpha, batch_size=batch_size)
                        clf.load_weights(best_weights)
                        print()

                        acc, f1 = clf.evaluate(test._Xtest, test._ytest, src=False)
                        print('acc: {0:.3f}'.format(acc))
                        print('f1:  {0:.3f}'.format(f1))

                        # print both gold and prediction to file for later
                        print_gold(test._ytest, os.path.join(args.predictions_dir, '{0}.gold.txt'.format(test_name)))
                        clf.evaluate(test._Xtest, test._ytest, src=False, outfile=os.path.join(args.predictions_dir, test_name, '{0}-epochs{1}-batch{2}-alpha{3}.txt'.format(tr_name, best_params[0], best_params[1], best_params[2])))
