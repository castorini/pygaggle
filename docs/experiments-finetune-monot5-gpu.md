# PyGaggle: Finetuning MonoT5 using Pytorch and GPU on [MS MARCO Passage Retrieval Small](https://github.com/microsoft/MSMARCO-Passage-Ranking)

This page contains instructions for finetuning MonoT5 models on the MS MARCO *passage* ranking task.
Note that there is also a separate [MS MARCO *document* ranking task](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-doc.md).

After finetuning the model, we provide a reranking script and steps to reproduce similar results as in the [MonoT5 paper](https://arxiv.org/pdf/2003.06713.pdf).

Note 1: Run the following instructions at root of this repo.
Note 2: Make sure that you have access to a GPU with at least 14GB RAM.
Note 3: Installation must have been done from source and make sure the [anserini-eval](https://github.com/castorini/anserini-eval) submodule is pulled. 
To do this, first clone the repository recursively.

```
git clone --recursive https://github.com/castorini/pygaggle.git
```

Then install PyGaggle using:

```
pip install --upgrade pygaggle/
```

## Model

+ monoT5-base: Document Ranking with a Pretrained Sequence-to-Sequence Model [(Nogueira et al., 2020)](https://arxiv.org/pdf/2003.06713.pdf)

## MonoT5 finetuning
### Data Prep

We're first going to download and extract the triples.small from MS MARCO.
```
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz -P triples
tar -zxf triples/triples.train.small.tar.gz -C triples/
```
Then, create a new directory so the training script can save model checkpoints to.
```
mkdir monoT5_model/
```
### Train
To train the MonoT5 model, run the following command:
```
nohup python -um pygaggle.run.finetune_monot5 --triples_path triples/triples.train.small.tsv \
                                              --save_every_n_steps 10000 \
                                              --output_model_path monoT5_model/ &
tail -f nohup.out
```
Here, we consider one epoch to be 6.4e5 of lines of the triples.train.small.tsv, which corresponds to 12.8e5 training examples (positives + negatives) and 10k steps using batch of 128. The number of epochs defaults to 10 (100k steps). Adding the "--epoch" argument with an integer will include `n * 6.4e5` lines to the training set.
If you don't wish to save checkpoints every N step, simply omit the --save_every_n_steps argument.

With an Nvidia V100, it takes around 2~3 days to train the model for 10 epochs.
## Re-Ranking on MS MARCO dev set
### Data Prep
Let's download and extract the collection and dev set from MS MARCO:
```
mkdir runs
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz -P devset
tar -zxf devset/collectionandqueries.tar.gz -C devset/
```
And download the official MS MARCO evaluation script.
```
wget https://github.com/microsoft/MSMARCO-Passage-Ranking/blob/master/ms_marco_eval.py
```
### Re-Rank
Prior to running this, we suggest looking at our first-stage [BM25 ranking instructions](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md) to generate the BM25 run in MS MARCO format and making sure the results are the same. Place it under `runs/` directory.
```
#####################
MRR @10: 0.18741227770955546
QueriesRanked: 6980
#####################
```
Let's now rerank the BM25 run using the previous trained model:
```
nohup python -um pygaggle.run.evaluate_monot5_reranker --model_name_or_path monoT5_model/ \
                                              --initial_run runs/run.msmarco-bm25-dev-passage.txt \
                                              --corpus devset/collection.tsv \
                                              --queries devset/queries.dev.small.tsv \
                                              --output_run runs/run.monot5-dev &
tail -f nohup
```
Your corpus can either be in .tsv or .jsonl format. The reranking should take about half a day on a V100.
If you check the `runs/` directory after it's done, there are two newly re-ranked runs: one in MS MARCO format and another in TREC.
To measure MRR@10 on the dev set, type:
```
python ms_marco_eval.py devset/qrels.dev.small.tsv runs/run.monot5-dev-marco.txt
```
This should yield around:
```
#####################
MRR @10: 0.3761892823031791
QueriesRanked: 6980
#####################
```

You can also try modifying the [finetune_monot5.py](https://github.com/castorini/pygaggle/blob/master/pygaggle/run/finetune_monot5.py) script training parameters to your liking.

If you were able to replicate these results, please submit a PR adding to the replication log!

