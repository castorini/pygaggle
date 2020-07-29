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

## Additional Instructions

0. Clone the repo with `git clone --recursive https://github.com/castorini/pygaggle.git`

0. Make you sure you have an installation of [Python 3.6+](https://www.python.org/downloads/). All `python` commands below refer to this.

0. For pip, do `pip install -r requirements.txt`
    * If you prefer Anaconda, use `conda env create -f environment.yml && conda activate pygaggle`.


# A simple reranking example - T5
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
# scores = [-1.004280924797058, -6.026238441467285]
```

# A simple reranking example - BERT
You can also try the code below, which uses a BERT reranker from [Passage Re-ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf).
Note that the T5 reranker produces slightly better scores than the BERT reranker.
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pygaggle.model import BatchTokenizer
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import SequenceClassificationTransformerReranker

model_name = 'castorini/monobert-large-msmarco'
tokenizer_name = 'bert-large-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
reranker =  SequenceClassificationTransformerReranker(model, tokenizer)

query = Query('what causes low liver enzymes')

correct_doc = Text('Reduced production of liver enzymes may indicate dysfunction of the liver. This article explains the causes and symptoms of low liver enzymes. Scroll down to know how the production of the enzymes can be accelerated.')

wrong_doc = Text('Elevated liver enzymes often indicate inflammation or damage to cells in the liver. Inflamed or injured liver cells leak higher than normal amounts of certain chemicals, including liver enzymes, into the bloodstream, elevating liver enzymes on blood tests.')

documents = [correct_doc, wrong_doc]

scores = [result.score for result in reranker.rerank(query, documents)]
# scores = [-3.077077865600586, -5.45782470703125]
```
