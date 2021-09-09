# Neural Pointwise Ranking Baselines on Robust04 - with GPU

This page contains instructions for running monoT5 on the Robust 04 collection using GPUs.

To learn more about monoT5, please read "Document Ranking with a Pretrained Sequence-to-Sequence Model" [(Nogueira et al., 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.63.pdf)


**Note**: Robust04 uses [TREC Disks 4 & 5](https://trec.nist.gov/data/cd45/index.html) corpora, which are only available after filling and signing a release form from NIST. Therefore, only proceed with this documentation if you already have the corpus.

We will focus on using monoT5-base since it is difficult to run such a large model without a TPU.

Prior to running this, we suggest looking at our first-stage [BM25 ranking instructions](https://github.com/castorini/anserini/blob/master/docs/regressions-robust04.md).

We rerank the BM25 run files that contain ~1,000 documents per query using monoT5.

MonoT5 is a pointwise reranker. This means that each document is scored independently using T5.

Note that we do not train monoT5 on Robust04. Hence, the results are **zero-shot**.


## Data Prep
We store all the files in the `data/robust04` directory:
```
export DATA_DIR=data/robust04
mkdir ${DATA_DIR}
```

We download the query, qrels and corpus files. The run file was generated during the BM25 stage and contains ~1,000 documents per query.

You can change the amount of candidate texts by setting the -hits parameter when performing Anserini's [BM25 ranking instructions](https://github.com/castorini/anserini/blob/master/docs/regressions-robust04.md).

In short, the files are:
- `topics.robust04.txt`: 250 queries (also called "topics") from Robust04.
- `qrels.robust04.txt`: 311,410 pairs of query and relevant document ids.
- `trec_disks_4_and_5_concat.txt`: TREC disks 4 & 5 documents (528,164) concatenated as a single text file.
- `run.bm25.txt`: 242,339 pairs of queries and retrieved documents using Anserini's BM25 (1,000 hits).

Let's start:
```
cd ${DATA_DIR}
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.robust04.txt
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.robust04.txt
wget https://storage.googleapis.com/castorini/robust04/trec_disks_4_and_5_concat.txt
cd ../../
```

If not generated yet, you can download the run file (1,000 hits):
```
wget https://storage.googleapis.com/castorini/robust04/run.robust04.bm25.txt
```

As a sanity check, we can evaluate the first-stage (BM25) retrieved documents using the `trec_eval` tool:
```
tools/eval/trec_eval.9.0.4/trec_eval -m map -m ndcg_cut.20 ${DATA_DIR}/qrels.robust04.txt ${DATA_DIR}/run.robust04.bm25.txt
```

The output should be:
```
map                     all     0.2531
ndcg_cut_20             all     0.4240
```

## Rerank with monoT5
We use the script below to prepare the query-doc pairs in the monoT5 input format and then rerank it using a monoT5-base model available in pygaggle:

```
python ./pygaggle/run/robust04_reranker_pipeline_gpu.py \
      --queries=${DATA_DIR}/topics.robust04.txt \
      --run=${DATA_DIR}/run.robust04.bm25.txt \
      --corpus=${DATA_DIR}/trec_disks_4_and_5_concat.txt \
      --output_monot5=${DATA_DIR}/monot5_results.txt \
      >> ${DATA_DIR}/output.log 2>&1
```
You might want to run this process in background using `screen` to make sure it does not get killed.

Using a NVIDIA Tesla T4, it takes approximately 16 hours to rerank with monoT5-base for ~1,000 candidate texts per query.
It creates an output file containing monoT5 output:

- `monot5_results.txt`

Each line in the output follows `trec_eval` format:
```
f'{query_id} Q0 {docid} {rank + 1} {1 / (rank + 1)} T5\n'
```

## Evaluate reranked results
After reranking is done, we can evaluate the reranked results using the `trec_eval` tool:

```
tools/eval/trec_eval.9.0.4/trec_eval -m map -m ndcg_cut.20 $DATA_DIR/qrels.robust04.txt $DATA_DIR/monot5_results.txt
```

For monoT5-base, the output should be:

```
map                   	all	0.3489
ndcg_cut_20           	all	0.5578
```

Note: These results are slightly higher than the ones obtained with TPUs, probably because we used a more recent version of spacy ('3.0.6' instead of '2.2.4').

If you were able to replicate these results, please submit a PR adding to the replication log.
Please mention in your PR if you note any differences.

## Replication Log

