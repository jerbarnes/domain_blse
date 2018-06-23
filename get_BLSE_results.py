from BLSE_domain import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json


def get_best_dev(models_dir):
    results = {}
    for d in os.listdir(models_dir):
        results[d] = {}
        for test in ['books', 'dvd', 'electronics', 'kitchen', 'semeval_2013', 'semeval_2016']:
            try:
                f1, params, weights = get_best_run(os.path.join(models_dir, d, test))
                results[d][test] = (f1, params, weights)
            except FileNotFoundError:
                pass
    return results


semeval_2013 = Semeval_Dataset('datasets/semeval_2013', None,
                                      binary=True, rep=words,
                                      one_hot=False)

semeval_2016 = Semeval_Dataset('datasets/semeval_2016', None,
                                      binary=True, rep=words,
                                      one_hot=False)

books = Book_Dataset(None, rep=words, one_hot=False, binary=True)
dvd = DVD_Dataset(None, rep=words, one_hot=False, binary=True)
electronics = Electronics_Dataset(None, rep=words, binary=True, one_hot=False)
kitchen = Kitchen_Dataset(None, rep=words, binary=True, one_hot=False)

dev_results = get_best_dev('models/ppmi-pivots/')

results = {}

data =  [    ('semeval_2016', semeval_2016, '/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS/twitter_embeddings.txt'),
             ('semeval_2013', semeval_2013, '/home/jeremy/NS/Keep/Temp/Exps/EMBEDDINGS/twitter_embeddings.txt'),
             ('books', books, 'embeddings/amazon/amazon-sg-300.txt'),
             ('dvd', dvd, 'embeddings/amazon/amazon-sg-300.txt'),
             ('electronics', electronics, 'embeddings/amazon/amazon-sg-300.txt'),
             ('kitchen', kitchen, 'embeddings/amazon/amazon-sg-300.txt')]

for tr_name, train, source_embedding_file in data:
    results[tr_name] = {}
    print('Train {0}'.format(tr_name))
    for test_name, test, target_embedding_file in data:
        if tr_name != test_name:
            try:
                src = WordVecs(source_embedding_file)
                trg = WordVecs(target_embedding_file)
                blse = BLSE_domain(src, trg, None, None, None, 2)
                results[tr_name][test_name] = {}
                results[tr_name][test_name]['BLSE'] = {}
                dev_f1, params, weights = dev_results[tr_name][test_name]
                blse.load_weights(weights)
                pred = blse.predict(test._Xtest, src=False).data.numpy().argmax(1)
                acc = accuracy_score(pred, test._ytest)
                prec = precision_score(pred, test._ytest)
                recall = recall_score(pred, test._ytest)
                f1 = per_class_f1(test._ytest, pred).mean()
                print('{0}: {1:.3f}'.format(test_name, f1))
                # print predictions
                filename = 'predictions/' + tr_name + '_to_' + test_name + 'f1:{0:.3f}'.format(f1)
                with open(filename, 'w') as out:
                    for line in pred:
                        out.write('{0}\n'.format(line))
                results[tr_name][test_name]['BLSE']['dev_f1'] = dev_f1
                results[tr_name][test_name]['BLSE']['params'] = params
                results[tr_name][test_name]['BLSE']['acc'] = acc
                results[tr_name][test_name]['BLSE']['prec'] = prec
                results[tr_name][test_name]['BLSE']['rec'] = recall
                results[tr_name][test_name]['BLSE']['f1'] = f1
            except FileNotFoundError:
                pass
            except KeyError:
                pass


with open('results/blse.txt', 'w') as out:
    json.dump(results, out)
