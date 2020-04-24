# PyGaggle

PyGaggle is a text ranking and question answering library using deep neural architectures.
It was designed for tight integration with [Pyserini](http://pyserini.io/), but can be easily adapted for other sources as well.

Currently, we provide a gaggle of rerankers for [CovidQA](https://github.com/castorini/pygaggle/blob/master/data/) on CORD-19 &mdash; the models described in ["Rapidly Bootstrapping a Question Answering Dataset for COVID-19"](https://arxiv.org/abs/2004.11339).

## Installation

1. For pip, do `pip install pygaggle`. If you prefer Anaconda, use `conda env create -f environment.yml && conda activate pygaggle`.

2. Install [PyTorch 1.4+](http://pytorch.org/).

3. Download the index: `sh scripts/update-index.sh`.

4. Make sure you have an installation of Java 11+: `javac --version`.

5. Install [Anserini](https://github.com/castorini/anserini).


## Running rerankers on CovidQA

By default, the script uses `data/lucene-index-covid-paragraph` for the index path.
If this is undesirable, set the environment variable `CORD19_INDEX_PATH` to the path of the index.


### Unsupervised Methods

**BM25**:

```bash
python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25
```

**BERT**:

```python
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

Instructions for our other MARCO and SQuAD models coming soon.
