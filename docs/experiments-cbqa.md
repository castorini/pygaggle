# PyGaggle: Dense Passage Retrieval (DPR) Baselines

This page contains instructions for running the Closed Book Question Answering.

First install Pygaggle

```
git clone --recursive https://github.com/castorini/pygaggle.git
```

Then

```
pip install pygaggle/
```

## Models

+ Dense Passage Retrieval for Open-Domain Question Answering [(Karpukhin et al., 2020)](https://arxiv.org/pdf/2004.04906.pdf)

## Natural Questions (NQ)
We first download the retrieval results from Pyserini:
```bash
wget https://www.dropbox.com/s/flby7nmthaqbzxo/run.dpr.nq-test.single.bf.json -P data
```

Then we run the inference and evaluation:
```bash
python -um pygaggle.run.evaluate_closed_book_question_answering --data data/run.dpr.nq-test.single.bf.json \
                                                                --model-name google/t5-large-ssm-nq \
                                                                --device cuda:0
```

The following output will be visible after it has finished:
```
Exact Match Accuracy: 28.89196675900277
```

If you were able to replicate these results, please submit a PR adding to the replication log!
Please mention in your PR if you find any difference!


## Replication Log

+ Results replicated by [@AlexWang000](https://github.com/AlexWang000) on 2021-11-01 (commit[`3906bae`](b45841a06f0240a0f7f8c8eb0c04700e79ad9ef6)) (RTX 3080)
