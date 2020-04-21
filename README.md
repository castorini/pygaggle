# PyGaggle

A gaggle of CORD-19 rerankers.

## Installation

1. `conda env create -f environment.yml && conda activate pygaggle`

2. Install [PyTorch 1.4+](http://pytorch.org/).

3. Download the index: `sh scripts/update-index.sh`

4. Make sure you have an installation of Java 8+: `javac --version` 


## Evaluating Highlighters

Run `sh scripts/evaluate-highlighters.sh`.