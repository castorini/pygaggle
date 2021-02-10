# PyGaggle: Dense Passage Retrieval (DPR) Baselines

This page contains instructions for running the Dense Passage Retrieval tools.

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
Note: On Compute Canada, you may have to install tensorflow separately by the following command.
```
pip install tensorflow_gpu 
```

## Models

+ Dense Passage Retrieval for Open-Domain Question Answering [(Karpukhin et al., 2020)](https://arxiv.org/pdf/2004.04906.pdf)

## Data Prep

We download the queries, top passages from the retriever, and ground truth answers. We download the examples in `data/` using
```
cd data/
wget https://www.dropbox.com/s/5izln5ws7aqn8am/retrieval_dpr_50.json
```

As a sanity check, the MD5sum of this file is `d0dd6edc9f7ac77956b5f9998b803fd2`.

Now, let's evaluate the DPR Reader.

```
cd ../
python -um pygaggle.run.evaluate_passage_reader --task wikipedia \
                                                --method dpr \ 
                                                --retrieval-file data/retrieval_dpr_50.json \
                                                --output-file my_dpr_prediction.json \
                                                --use-top-k-passages 10
```

The final Exact Match (EM) score after 8757 queries should be `39.328537170263786`.

It should take about 1-2 hours to read and answer all of the queries on a GPU.
The type of GPU will directly influence your inference time.

If you were able to replicate these results, please submit a PR adding to the replication log!
Please mention in your PR if you find any difference!


## Replication Log

+ Results replicated by [@KaiSun314](https://github.com/KaiSun314) on 2021-02-05 (commit[`de20456`](https://github.com/castorini/pygaggle/commit/de2045635106abf104cd7adaf07525fd4f173a3b)) (Tesla V100 on Compute Canada)
