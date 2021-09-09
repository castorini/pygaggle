# PyGaggle: Dense Passage Retrieval (DPR) Baselines

This page contains instructions for running the Dense Passage Retrieval tools.

Note: If you're using Compute Canada clusters for replicating this experiment, then do 

```
export PYTHONPATH=
export PIP_CONFIG_FILE=
``` 

First, clone the PyGaggle repository recursively.

```
git clone --recursive https://github.com/castorini/pygaggle.git
cd pygaggle/
```
Then load Java module:
```
module load java
```
Then install Pytorch.
```
pip install torch
```
Then install PyGaggle by the following command.
```
pip install -r requirements.txt
```
Note: To run the commands below on a Compute Canada cluster, make sure you're on a compute node as login node won't have enough memory to support these operations and you may get segmentation faults trying to run these. If GPU resources are not allocated, then commands can be run with the flag `--device cpu`

## Models

+ Dense Passage Retrieval for Open-Domain Question Answering [(Karpukhin et al., 2020)](https://arxiv.org/pdf/2004.04906.pdf)

## Natural Questions (NQ)
End-to-end answer prediction using **DPR Retrieval**

We first download the retrieval results from Pyserini:
```bash
$ wget https://www.dropbox.com/s/flby7nmthaqbzxo/run.dpr.nq-test.single.bf.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.55 \
                                                             gar garfusion_0.43_0.3053 \
                                                  --retrieval-file data/run.dpr.nq-test.single.bf.json --topk-em 40 80 180 200
```

The following output will be visible after it has finished:
```
Setting: DPR
Top40       Exact Match Accuracy: 41.19113573407202
Setting: DPR Fusion, beta=1.0, gamma=0.55
Top200      Exact Match Accuracy: 42.54847645429363
Setting: GAR
Top80       Exact Match Accuracy: 41.52354570637119
Setting: GAR Fusion, beta=0.43, gamma=0.3053
Top180      Exact Match Accuracy: 43.46260387811635
```

End to end answer prediction using **BM25 Retrieval**


We first download the retrieval results from Pyserini:
```bash
$ wget https://www.dropbox.com/s/0ufwszz6son8e2j/run.dpr.nq-test.bm25.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.65 \
                                                             gar garfusion_0.46_0.308 \
                                                  --retrieval-file data/run.dpr.nq-test.bm25.json --topk-em 500
```

The following output will be visible after it has finished:

```
Setting: DPR
Top500      Exact Match Accuracy: 36.31578947368421
Setting: DPR Fusion, beta=1.0, gamma=0.65
Top500      Exact Match Accuracy: 36.95290858725762
Setting: GAR
Top500      Exact Match Accuracy: 37.285318559556785
Setting: GAR Fusion, beta=0.46, gamma=0.308
Top500      Exact Match Accuracy: 38.337950138504155
```

End to end answer prediction using **Hybrid Retrieval**

We first download the retrieval results from Pyserini:
```bash
$ wget https://www.dropbox.com/s/35l5hbtie7gztjf/run.dpr.nq-test.single.bf.bm25.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.63 \
                                                             gar garfusion_0.32_0.1952 \
                                                  --retrieval-file data/run.dpr.nq-test.single.bf.bm25.json \
                                                  --topk-em 20 180 200 \
                                                  --batch-size 10
```
The following output will be visible after it has finished:

```
Setting: DPR
Top20	Exact Match Accuracy: 41.24653739612189
Setting: DPR Fusion, beta=1.0, gamma=0.63
Top200	Exact Match Accuracy: 43.15789473684211
Setting: GAR
Top20	Exact Match Accuracy: 41.883656509695285
Setting: GAR Fusion, beta=0.32, gamma=0.1952
Top180	Exact Match Accuracy: 44.01662049861496
```

## TriviaQA 
End to end answer prediction using **DPR Retrieval**

We first download the retrieval results from Pyserini:
```bash
$ wget https://www.dropbox.com/s/376e4y5pe00symw/run.dpr.trivia-test.multi.bf.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.18 \
                                                             gar garfusion_1.18_0.2478 \
                                                  --model-name facebook/dpr-reader-multiset-base \
                                                  --retrieval-file data/run.dpr.trivia-test.multi.bf.json --topk-em 440 480 500
```

The following output will be visible after it has finished:

```
Setting: DPR
Top508	Exact Match Accuracy: 57.509060373022194
Setting: DPR Fusion, beta=1.0, gamma=0.18
Top480	Exact Match Accuracy: 58.3488022628834
Setting: GAR
Top440	Exact Match Accuracy: 58.91452311500044
Setting: GAR Fusion, beta=1.18, gamma=0.2478
Top500	Exact Match Accuracy: 59.4890833554318
```

End to end answer prediction using **BM25 Retrieval**

We first download the retrieval results from Pyserini:
```bash
$ wget https://www.dropbox.com/s/vaoyo1zq3h8har5/run.dpr.trivia-test.bm25.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.45 \
                                                             gar garfusion_0.78_0.093 \
                                                  --model-name facebook/dpr-reader-multiset-base \
                                                  --retrieval-file data/run.dpr.trivia-test.bm25.json --topk-em 480 500
```

The following output will be visible after it has finished:

```
Setting: DPR
Top480      Exact Match Accuracy: 58.84380800848581
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Top480      Exact Match Accuracy: 58.83496862017148
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Setting: DPR Fusion, beta=1.0, gamma=0.45
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Top480      Exact Match Accuracy: 59.223901706001946
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Setting: GAR
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Top500      Exact Match Accuracy: 61.11553080526827
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Setting: GAR Fusion, beta=0.78, gamma=0.093
2021-04-11 22:09:42 [INFO] evaluate_passage_reader: Top480      Exact Match Accuracy: 61.575178997613364
```

End to end answer prediction using **Hybrid Retrieval**

We first download the retrieval results from Pyserini
```bash
$ wget https://www.dropbox.com/s/7hp04w71o4eravc/run.dpr.trivia-test.multi.bf.bm25.json -P data
```

Then we run the inference and evaluation:
```bash
$ python -um pygaggle.run.evaluate_passage_reader --task wikipedia --retriever score --reader dpr \
                                                  --settings dpr dprfusion_1.0_0.17 \
                                                             gar garfusion_0.76_0.152 \
                                                  --model-name facebook/dpr-reader-multiset-base \
                                                  --retrieval-file data/run.dpr.trivia-test.multi.bf.bm25.json --topk-em 40 160 260 460
```

The following output will be visible after it has finished:

```
Setting: DPR
Top40       Exact Match Accuracy: 59.108989657915664
Setting: DPR Fusion, beta=1.0, gamma=0.17
Top460      Exact Match Accuracy: 59.98408910103421
Setting: GAR
Top160      Exact Match Accuracy: 61.035976310439324
Setting: GAR Fusion, beta=0.76, gamma=0.152
Top260      Exact Match Accuracy: 61.73428798727129
```

If you were able to replicate these results, please submit a PR adding to the replication log!
Please mention in your PR if you find any difference!


## Replication Log

+ Results replicated by [@MXueguang](https://github.com/MXueguang) on 2021-04-11 (commit[`1c1fb64`](1c1fb644ec7bca65a507ed2cc3a1ada21a2a5976)) (RTX 2080)
+ Results replicated by [@mayankanand007](https://github.com/mayankanand007) on 2021-08-16 (commit[`dda194f`](dda194f4546af0db62c317bbaf5ccd58edae0591))
