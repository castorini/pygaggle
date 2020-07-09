# PyGaggle: Neural Ranking Baselines on CovidQA

This page contains instructions for running various neural reranking baselines on the CovidQA ranking task. 

Note 1: Run the following instructions at root of this repo.
Note 2: Make sure that you have access to a GPU
Note 3: Installation must have been done from source and make sure the [anserini-eval](https://github.com/castorini/anserini-eval) submodule is pulled. 
To do this, first clone the repository recursively.

```
git clone --recursive https://github.com/castorini/pygaggle.git
```

Then install PyGaggle using:

```
pip install pygaggle/
```

## Re-Ranking with Random

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method random \
                                                    --dataset data/kaggle-lit-review-0.2.json \
                                                    --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1     0.0
recall@3        0.0199546485260771
recall@50       0.3247165532879819
recall@1000     1.0
mrr     0.03999734528458418
mrr@10  0.020888672929489253
```

## Re-Ranking with BM25

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 \
                                                    --dataset data/kaggle-lit-review-0.2.json \
                                                    --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1     0.14685314685314685
recall@3        0.2199546485260771
recall@50       0.6582766439909296
recall@1000     0.6820861678004534
mrr     0.24651188194041115
mrr@10  0.2267060792570997
```

It takes about 10 seconds to re-rank this subset on CovidQA using a P100.

## Re-Ranking with monoT5-Base

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5 \
                                                    --dataset data/kaggle-lit-review-0.2.json \
                                                    --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1     0.2789115646258503
recall@3        0.41854551344347257
recall@50       0.92555879494655
recall@1000     1.0
mrr     0.417982565405279
mrr@10  0.4045405463772811
```

It takes about 17 minutes to re-rank this subset on CovidQA using a P100.