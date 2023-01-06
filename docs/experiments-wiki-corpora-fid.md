# PyGaggle: Reproducing FiD Results With Improved Wikipedia Corpus Variants

Fusion-in-Decoder (FiD) is a model described in the following paper:

> Izacard, Gautier, and Édouard Grave. [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://aclanthology.org/2021.eacl-main.74/). _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_. 2021.

 We have replicated FiD training with our Wikipedia corpus variants, enabling the reproduction of these results in PyGaggle.

Our own efforts are described in the paper entitled: 
> "Pre-Processing Matters! Improved Wikipedia Corpora for Open-Domain Question Answering"

Our efforts include both retrieval as well as end-to-end answer generation.
We cover only end-to-end answer generation here; for retrieval, please see [this guide](https://github.com/castorini/pyserini/blob/master/docs/experiments-wiki-corpora.md) in our Pyserini IR library, which must be done first. Here we provide instructions to reproduce end-to-end answer generation on the ```wiki-all-6-3``` corpus variant with the NaturalQuestions and TriviaQA datasets.

## Setup
Make sure you have PyGaggle and Pyserini installed.

```bash
pip install pyserini
git clone https://github.com/castorini/pygaggle.git 
cd pygaggle
pip install --editable .
pip install -r requirements.txt
```

Now clone the FiD repo in the same directory as where PyGaggle was cloned.

```bash
git clone https://github.com/facebookresearch/FiD.git
cd FiD
pip install -r requirements.txt
pip install transformers==4.10.0
```

We make the FiD-large models available in HuggingFace🤗 for all corpus variants. The links to the models for wiki-all-6-3 are:

[wiki-all-6-3-fid-large-nq-reader](https://huggingface.co/castorini/wiki-all-6-3-fid-large-nq-reader)  
[wiki-all-6-3-fid-large-tqa-reader](https://huggingface.co/castorini/wiki-all-6-3-fid-large-tqa-reader)  

To download the models for the next step, first navigate to the ```FiD/pretrained_models/``` directory

Make sure you have Git LFS set up by running 
```
git lfs install
```
Then to download, run:
```
git clone https://huggingface.co/castorini/wiki-all-6-3-fid-large-nq-reader
git clone https://huggingface.co/castorini/wiki-all-6-3-fid-large-tqa-reader
```

## Inference and Evaluation

Then we run inference and evaluation in the top level directory of PyGaggle. The retrieval_file will be from the hybrid retrieval step of the [Pyserini guide](https://github.com/castorini/pyserini/blob/master/docs/experiments-wiki-corpora.md).

### Natural Questions
```
$ python3 -um pygaggle.run.evaluate_fid_reader \
    --model_name wiki-all-6-3-fid-large-nq-reader \
    --tokenizer-name t5-large \
    --retrieval_file runs/run.wiki-all-6-3.nq-test.hybrid.json \
    --output_file runs/run.wiki-all-6-3.nq-test.hybrid.FiD-output.json
```
The expected output by the script at the end is:
```
[INFO] evaluate_fid_reader: Exact Match Accuracy: 55.81717451523546
```
Your output will hopefully mostly match

### TriviaQA
```
$ python3 -um pygaggle.run.evaluate_fid_reader \
    --model_name wiki-all-6-3-fid-large-tqa-reader \
    --tokenizer-name t5-large \
    --retrieval_file runs/run.wiki-all-6-3.dpr-trivia-test.hybrid.json \
    --output_file runs/run.wiki-all-6-3.dpr-trivia-test.hybrid.FiD-output.json
```
The expected output by the script at the end is:
```
[INFO] evaluate_fid_reader: Exact Match Accuracy: 73.72933792981526
```
Your output will hopefully mostly match

## Reproduction Log


