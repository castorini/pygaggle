# PyGaggle

[![PyPI](https://img.shields.io/pypi/v/pygaggle?color=brightgreen)](https://pypi.org/project/pygaggle/)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

PyGaggle provides a gaggle of deep neural architectures for text ranking and question answering.
It was designed for tight integration with [Pyserini](http://pyserini.io/), but can be easily adapted for other sources as well.

Currently, this repo contains implementations of the rerankers for [CovidQA](https://github.com/castorini/pygaggle/blob/master/data/) on CORD-19, as described in ["Rapidly Bootstrapping a Question Answering Dataset for COVID-19"](https://arxiv.org/abs/2004.11339).

## Installation

1. For pip, do `pip install pygaggle`. If you prefer Anaconda, use `conda env create -f environment.yml && conda activate pygaggle`.

2. Install [PyTorch 1.4+](http://pytorch.org/).

3. Download the index: `sh scripts/update-index.sh`.

4. Make sure you have an installation of Java 11+: `javac --version`.

5. Install [Anserini](https://github.com/castorini/anserini).


## Running rerankers on CovidQA

By default, the script uses `data/lucene-index-covid-paragraph` for the index path.
If this is undesirable, set the environment variable `CORD19_INDEX_PATH` to the path of the index.
For a full list of mostly self-explanatory environment variables, see [this file](https://github.com/castorini/pygaggle/blob/master/pygaggle/settings.py#L7).

BM25 uses the CPU. If you don't have a GPU for the transformer models, pass `--device cpu` (PyTorch device string format) to the script.


### Unsupervised Methods

**BM25**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25
```

**BERT**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name bert-base-cased
```

**SciBERT**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name allenai/scibert_scivocab_cased
```

**BioBERT**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name biobert
```

### Supervised Methods

**T5 (fine-tuned on MS MARCO)**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method t5
```

**BioBERT (fine-tuned on SQuAD v1.1)**:

1. Download the weights, vocab, and config from the [BioBERT repository](https://github.com/dmis-lab/bioasq-biobert) to the same folder.

2. Rename the following files in the folder:

```bash
mv bert_config.json config.json
for filename in model.ckpt*; do 
    mv $filename $(python -c "import re; print(re.sub(r'ckpt-\\d+', 'ckpt', '$filename'))");
done
```

3. Evaluate the model:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method qa_transformer --model-name <folder path>
```

**BioBERT (fine-tuned on MS MARCO)**:

1. Download the weights, vocab, and config from our Google Storage bucket. This requires an installation of [gsutil](https://cloud.google.com/storage/docs/gsutil).

```bash
mkdir biobert-marco && cd biobert-marco
gsutil cp "gs://neuralresearcher_data/doc2query/experiments/exp374/model.ckpt-100000*" .
gsutil cp gs://neuralresearcher_data/biobert_models/biobert_v1.1_pubmed/bert_config.json config.json
gsutil cp gs://neuralresearcher_data/biobert_models/biobert_v1.1_pubmed/vocab.txt .
```

2. Rename the files:

```bash
for filename in model.ckpt*; do 
    mv $filename $(python -c "import re; print(re.sub(r'ckpt-\\d+', 'ckpt', '$filename'))");
done
```

3. Evaluate the model:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method seq_class_transformer --model-name <folder path>
```
