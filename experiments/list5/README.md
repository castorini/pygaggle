# LisT5: FEVER Pipeline with T5

This page describes replication for the LisT5 pipeline for fact verification, outlined in the following paper:
* Kelvin Jiang, Ronak Pradeep, Jimmy Lin. [Exploring Listwise Evidence Reasoning with T5 for Fact Verification.](https://aclanthology.org/2021.acl-short.51.pdf) _ACL 2021_.

Some initial setup:

```bash
mkdir data/list5
mkdir runs/list5
```

## Document Retrieval

1. Retrieve with anserini

Follow instructions [here](https://github.com/castorini/anserini/blob/master/docs/experiments-fever.md) to download the FEVER dataset and build an index with anserini. Assume the anserini directory is located at `~/anserini`. After the dataset has been indexed, run the following command to retrieve (note the use of the paragraph index):

```bash
sh ~/anserini/target/appassembler/bin/SearchCollection \
 -index ~/anserini/indexes/fever/lucene-index-fever-paragraph \
 -topicreader TsvInt -topics ~/anserini/collections/fever/queries.sentence.test.tsv \
 -output runs/list5/run.fever-anserini-paragraph.test.tsv \
 -bm25 -bm25.k1 0.6 -bm25.b 0.5
```

2. Retrieve with MediaWiki API (UKP-Athene)

Also retrieve documents with MediaWiki API, code originating from [UKP-Athene's repository](https://github.com/UKPLab/fever-2018-team-athene). Install the dependencies for UKP-Athene's code using `experiments/list5/ukp-athene/requirements.txt` if necessary.

```bash
python experiments/list5/ukp-athene/doc_retrieval.py \
 --db-file ~/anserini/collections/fever/wiki-pages \
 --in-file ~/anserini/collections/fever/shared_task_test.jsonl \
 --out-file runs/list5/run.fever-ukp-athene-paragraph.test.jsonl
```

Convert the MediaWiki API results to run format.

```bash
python experiments/list5/ukp-athene/convert_to_run.py \
 --dataset_file runs/list5/run.fever-ukp-athene-paragraph.test.jsonl \
 --output_run_file runs/list5/run.fever-ukp-athene-paragraph.test.tsv
```

3. Merge retrieval runs

Merge the results of the two methods of retrieval into a single run. Make sure that the anserini run file comes first in the list of `--input_run_file` arguments.

```bash
python experiments/list5/merge_runs.py \
 --input_run_file runs/list5/run.fever-anserini-paragraph.test.tsv \
 --input_run_file runs/list5/run.fever-ukp-athene-paragraph.test.tsv \
 --output_run_file runs/list5/run.fever-paragraph.test.tsv \
 --strategy zip
```

## Sentence Selection

4. Expand document IDs to all sentence IDs

Expand run file to a sentence ID granularity.

```bash
python experiments/list5/expand_docs_to_sentences.py \
 --input_run_file runs/list5/run.fever-paragraph.test.tsv \
 --collection_folder ~/anserini/collections/fever/wiki-pages \
 --output_run_file runs/list5/run.fever-sentence-top-150.test.tsv \
 --k 150
```

5. Convert run file to T5 input file for monoT5 re-ranking

Re-rank the top `k = 200` sentences, a tradeoff between efficiency and recall.

```bash
python experiments/list5/convert_run_to_sentence_selection_input.py \
 --dataset_file ~/anserini/collections/fever/shared_task_test.jsonl \
 --run_file runs/list5/run.fever-sentence-top-150.test.tsv \
 --collection_folder ~/anserini/collections/fever/wiki-pages \
 --output_id_file data/list5/query-doc-pairs-id-test-ner-rerank-top-200.txt \
 --output_text_file data/list5/query-doc-pairs-text-test-ner-rerank-top-200.txt \
 --k 200 --type mono --ner
```

6. Re-rank T5 input file to get scores file

Run inference of T5 sentence selection model (e.g. using Google Cloud TPUs). Assume the sentence selection T5 output file is at `data/list5/query-doc-pairs-scores-test-ner-rerank-top-200.txt`.

7. Convert scores file back to run file

```bash
python experiments/list5/convert_sentence_selection_output_to_run.py \
 --id_file data/list5/query-doc-pairs-id-test-ner-rerank-top-200.txt \
 --scores_file data/list5/query-doc-pairs-scores-test-ner-rerank-top-200.txt \
 --output_run_file runs/list5/run.fever-sentence-top-150-reranked.txt \
 --type mono
```

## Label Prediction

8. Convert re-ranked run file to T5 input file for label prediction

Make sure to use `--format concat` to specify listwise format. 

```bash
python experiments/list5/convert_run_to_label_prediction_input.py \
 --dataset_file ~/anserini/collections/fever/shared_task_test.jsonl \
 --run_file runs/list5/run.fever-sentence-top-150-reranked.txt \
 --collection_folder ~/anserini/collections/fever/wiki-pages \
 --output_id_file data/list5/query-doc-pairs-id-test-ner-label-pred-concat.txt \
 --output_text_file data/list5/query-doc-pairs-text-test-ner-label-pred-concat.txt \
 --format concat
```

9. Predict labels for T5 input file to get scores file

Run inference of T5 label prediction model (e.g. using Google Cloud TPUs). Assume the label prediction T5 output file is at `data/list5/query-doc-pairs-scores-test-ner-label-pred-concat.txt`.

10. Convert scores file to FEVER submission file

```bash
python experiments/list5/predict_for_submission.py \
 --id_file data/list5/query-doc-pairs-id-test-ner-label-pred-concat.txt \
 --scores_file data/list5/query-doc-pairs-scores-test-ner-label-pred-concat.txt \
 --dataset_file ~/anserini/collections/fever/shared_task_test.jsonl \
 --output_predictions_file data/list5/predictions.jsonl
```
