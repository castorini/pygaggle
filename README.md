# PyGaggle

[![PyPI](https://img.shields.io/pypi/v/pygaggle?color=brightgreen)](https://pypi.org/project/pygaggle/)
[![LICENSE](https://img.shields.io/badge/license-Apache-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

PyGaggle provides a gaggle of deep neural architectures for text ranking and question answering.
It was designed for tight integration with [Pyserini](http://pyserini.io/), but can be easily adapted for other sources as well.

Currently, this repo contains implementations of the rerankers for [CovidQA](https://github.com/castorini/pygaggle/blob/master/data/) on CORD-19, as described in ["Rapidly Bootstrapping a Question Answering Dataset for COVID-19"](https://arxiv.org/abs/2004.11339).

## Installation

0. Install via PyPI `pip install pygaggle`. Requires [Python 3.6+](https://www.python.org/downloads/)

0. Install [PyTorch 1.4+](http://pytorch.org/).

0. Download the index: `sh scripts/update-index.sh`.

0. Make sure you have an installation of Java 11+: `javac --version`.

0. Install [Anserini](https://github.com/castorini/anserini).


# A simple reranking example
The code below exemplifies how to score two documents for a given query using a T5 reranker from [Document Ranking with a Pretrained
Sequence-to-Sequence Model](https://arxiv.org/pdf/2003.06713.pdf).
```python
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from pygaggle.model import T5BatchTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import T5Reranker

model_name = 'castorini/monot5-base-msmarco'
tokenizer_name = 't5-base'
batch_size = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5ForConditionalGeneration.from_pretrained(model_name)
model = model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer = T5BatchTokenizer(tokenizer, batch_size)
reranker =  T5Reranker(model, tokenizer)

query = Query('what causes low liver enzymes')

correct_doc = Text('Reduced production of liver enzymes may indicate dysfunction of the liver. This article explains the causes and symptoms of low liver enzymes. Scroll down to know how the production of the enzymes can be accelerated.')

wrong_doc = Text('Elevated liver enzymes often indicate inflammation or damage to cells in the liver. Inflamed or injured liver cells leak higher than normal amounts of certain chemicals, including liver enzymes, into the bloodstream, elevating liver enzymes on blood tests.')

documents = [correct_doc, wrong_doc]

scores = [result.score for result in reranker.rerank(query, documents)]
# scores = [-0.1782158613204956, -0.36637523770332336]
```

# Evaluations

## Additional Instructions

0. Clone the repo with `git clone --recursive https://github.com/castorini/pygaggle.git`

0. Make you sure you have an installation of [Python 3.6+](https://www.python.org/downloads/). All `python` commands below refer to this.

0. For pip, do `pip install -r requirements.txt`
    * If you prefer Anaconda, use `conda env create -f environment.yml && conda activate pygaggle`.


## Running rerankers on CovidQA

For a full list of mostly self-explanatory environment variables, see [this file](https://github.com/castorini/pygaggle/blob/master/pygaggle/settings.py#L7).

BM25 uses the CPU. If you don't have a GPU for the transformer models, pass `--device cpu` (PyTorch device string format) to the script.

*Note: Run the following evaluations at root of this repo.*

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

0. `mkdir biobert-squad && cd biobert-squad`

0. Download the weights, vocab, and config from the [BioBERT repository](https://github.com/dmis-lab/bioasq-biobert) to `biobert-squad`.

0. Untar the model and rename some files in `biobert-squad`:

```bash
tar -xvzf BERT-pubmed-1000000-SQuAD.tar.gz
mv bert_config.json config.json
for filename in model.ckpt*; do
    mv $filename $(python -c "import re; print(re.sub(r'ckpt-\\d+', 'ckpt', '$filename'))");
done
```

0. Evaluate the model:

```bash
cd .. # go to root of this of repo
python -um pygaggle.run.evaluate_kaggle_highlighter --method qa_transformer --model-name <folder path>
```

**BioBERT (fine-tuned on MS MARCO)**:

0. Download the weights, vocab, and config from our Google Storage bucket. This requires an installation of [gsutil](https://cloud.google.com/storage/docs/gsutil_install?hl=ru).

```bash
mkdir biobert-marco && cd biobert-marco
gsutil cp "gs://neuralresearcher_data/doc2query/experiments/exp374/model.ckpt-100000*" .
gsutil cp gs://neuralresearcher_data/biobert_models/biobert_v1.1_pubmed/bert_config.json config.json
gsutil cp gs://neuralresearcher_data/biobert_models/biobert_v1.1_pubmed/vocab.txt .
```

0. Rename the files:

```bash
for filename in model.ckpt*; do
    mv $filename $(python -c "import re; print(re.sub(r'ckpt-\\d+', 'ckpt', '$filename'))");
done
```

0. Evaluate the model:

```bash
cd .. # go to root of this repo
python -um pygaggle.run.evaluate_kaggle_highlighter --method seq_class_transformer --model-name <folder path>
```
