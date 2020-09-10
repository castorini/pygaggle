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

NL Question:

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
mrr             0.03999734528458418
mrr@10          0.020888672929489253
```

Keyword Query

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method random \
                                                     --split kq \
                                                     --dataset data/kaggle-lit-review-0.2.json \
                                                     --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1     0.0
recall@3        0.0199546485260771
recall@50       0.3247165532879819
recall@1000     1.0
mrr             0.03999734528458418
mrr@10          0.020888672929489253
```

## Re-Ranking with BM25

NL Question:

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 \
                                                    --dataset data/kaggle-lit-review-0.2.json \
                                                    --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1    0.15384615384615385
recall@3       0.21865889212827985
recall@50      0.7208778749595076
recall@1000    0.7582928409459021
mrr            0.25329970378011524
mrr@10         0.23344131303314977

```

Keyword Query:

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25 \
                                                     --split kq \
                                                     --dataset data/kaggle-lit-review-0.2.json \
                                                     --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1    0.15384615384615385
recall@3       0.21865889212827985
recall@50      0.7208778749595076
recall@1000    0.7582928409459021
mrr            0.25441237140238665
mrr@10         0.23493413238311195
```

It takes about 10 seconds to re-rank this subset on CovidQA

## Re-Ranking with monoT5

NL Question:

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
mrr             0.417982565405279
mrr@10          0.4045405463772811
```

Keyword Query:

```
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5 \
                                                     --split kq \
                                                     --dataset data/kaggle-lit-review-0.2.json \
                                                     --index-dir indexes/lucene-index-cord19-paragraph-2020-05-12
```

The following output will be visible after it has finished:

```
precision@1     0.24489795918367346
recall@3        0.38566569484936825
recall@50       0.9231778425655977
recall@1000     1.0
mrr             0.37988285486956513
mrr@10          0.3671336788683727
```

It takes about 17 minutes to re-rank this subset on CovidQA using a P100.  It is worth noting again that you might need to modify the batch size to best fit the GPU at hand (--batch-size={BATCH_SIZE}).

If you were able to replicate these results, please submit a PR adding to the replication log!


## Replication Log

+ Results replicated by [@justinborromeo](https://github.com/justinborromeo) on 2020-09-08 (commit[`94befbd`](https://github.com/castorini/pygaggle/commit/94befbd58b19c3e46d930e67797102bf174efd01)) (GTX960M)
