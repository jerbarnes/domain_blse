Projecting Embeddings for Domain Adaptation: Joint Modeling of Sentiment in Diverse Domains
==============

This is the source code from the COLING paper:

Jeremy Barnes, Roman Klinger, and Sabine Schulde im Walde. 2018. **Projecting Embeddings for Domain Adaptation: Joint Modeling of Sentiment in Diverse Domains**. In *Proceedings of COLING 2018 (to appear)*.


If you use the code for academic research, please cite the paper in question:
```
@inproceedings{Barnes2018domain,
    author={Barnes, Jeremy and Klinger, Roman and Schulte im Walde, Sabine},
    title={Projecting Embeddings for Domain Adaptation: Joint Modeling of Sentiment in Diverse Domains},
    booktitle = {Proceedings of COLING 2018, the 27th International Conference on Computational Linguistics},
    year = {2018},
    month = {August},
    address = {Santa Fe, USA},
    publisher = {Association for Computational Linguistics},
    language = {english}
    }
```


Requirements to run the experiments
--------
- Python 3
- NumPy
- sklearn [http://scikit-learn.org/stable/]
- pytorch [http://pytorch.org/]



Usage
--------

First, clone the repo:

```
git clone https://github.com/jbarnesspain/domain_blse
cd blse
```


Then, get the embeddings, either by training your own,
or by downloading the [pretrained embeddings](https://drive.google.com/open?id=1GpyF2h0j8K5TKT7y7Aj0OyPgpFc8pMNS) mentioned in the paper,
and put them in the 'embeddings' directory:


Run the domain_blse script:

```
python3 BLSE_domain_all.py
```

Finally, you can use the blse.py script which will automatically use the best hyperparameters found.

```
python3 blse.py
``` 


License
-------

Copyright (C) 2018, Jeremy Barnes

Licensed under the terms of the Creative Commons CC-BY public license
