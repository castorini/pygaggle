# PyGaggle

A gaggle of rerankers for [CovidQA](https://github.com/castorini/pygaggle/blob/master/data/) and CORD-19. 

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

**BM25**: `python -um pygaggle.run.evaluate_kaggle_highlighter --method bm25`

**BERT**: `python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name bert-base-cased`

**SciBERT**: `python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name allenai/scibert_scivocab_cased`

**BioBERT**: `python -um pygaggle.run.evaluate_kaggle_highlighter --method transformer --model-name biobert`


### Supervised Methods

**T5 (MARCO)**: `python -um pygaggle.run.evaluate_kaggle_highlighter --method t5`

Instructions for our other MARCO and SQuAD models coming soon.
