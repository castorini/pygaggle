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


## A Simple Reranking Example

Here's how to initalize the T5 reranker from [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://arxiv.org/pdf/2003.06713.pdf):

```python
# Initialize the reranker
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
```

Alternatively, here's the BERT reranker from [Passage Re-ranking with BERT](https://arxiv.org/pdf/1901.04085.pdf), which isn't as good as the T5 reranker:

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
```

Either way, continue with a complere reranking example:

```python
# Here's our query:
query = Query('who proposed the geocentric theory')

# Option 1: fetch some passages to rerank from MS MARCO with Pyserini
from pyserini.search import SimpleSearcher

searcher = SimpleSearcher('/path/to/msmarco/index/')

hits = searcher.search(query.text)

from pygaggle.rerank.base import hits_to_texts
texts = hits_to_texts(hits)

# Option 2: here's what Pyserini would have retrieved, hard-coded
passages = [['7744105', 'For Earth-centered it was  Geocentric Theory proposed by greeks under the guidance of Ptolemy and Sun-centered was Heliocentric theory proposed by Nicolas Copernicus in 16th century A.D. In short, Your Answers are: 1st blank - Geo-Centric Theory. 2nd blank - Heliocentric Theory.'], ['2593796', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.he geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.'], ['6217200', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.opernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.'], ['3276925', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.'], ['6217208', 'Copernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect.opernicus proposed a heliocentric model of the solar system â\x80\x93 a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.'], ['4280557', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.imple tools, such as the telescope â\x80\x93 which helped convince Galileo that the Earth was not the center of the universe â\x80\x93 can prove that ancient theory incorrect. You might want to check out one article on the history of the geocentric model and one regarding the geocentric theory.'], ['264181', 'Nicolaus Copernicus (b. 1473â\x80\x93d. 1543) was the first modern author to propose a heliocentric theory of the universe. From the time that Ptolemy of Alexandria (c. 150 CE) constructed a mathematically competent version of geocentric astronomy to Copernicusâ\x80\x99s mature heliocentric version (1543), experts knew that the Ptolemaic system diverged from the geocentric concentric-sphere conception of Aristotle.'], ['4280558', 'A Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth. Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth.'], ['3276926', 'The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.'], ['5183032', "After 1,400 years, Copernicus was the first to propose a theory which differed from Ptolemy's geocentric system, according to which the earth is at rest in the center with the rest of the planets revolving around it."]]

texts = [ Text(p[1], {'docid': p[0]}, 0) for p in passages] # Note, scores don't matter since T5 will ignore them.

# Either option, let's print out the passages prior to reranking:
for i in range(0, 10):
    print(f'{i+1:2} {texts[i].metadata["docid"]:15} {texts[i].score:.5f} {texts[i].text}')

# Finally, rerank:
reranked = reranker.rerank(query, texts)
reranked.sort(key=lambda x: x.score, reverse=True)

# Print out reranked results:
for i in range(0, 10):
    print(f'{i+1:2} {texts[i].metadata["docid"]:15} {reranked[i].score:.5f} {reranked[i].text}')
```
