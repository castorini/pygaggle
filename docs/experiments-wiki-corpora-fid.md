# PyGaggle: Reproducing FiD Results With Improved Wikipedia Corpus Variants

This page contains instructions for running the FiD tools. We have replicated FiD results with our Wikipedia corpus variants and incorporated the technique into Pygaggle.

Our own efforts are described in the paper entitled: "Pre-Processing Matters! Improved Wikipedia Corpora for Open-Domain Question Answering"

This guide provides instructions to reproduce some of the commands in our study. Note that you will need to change the parameters to the commands as necessary.
Our efforts include both retrieval as well as end-to-end answer generation.
We cover only end-to-end answer generation here; for retrieval, please see [this guide](https://github.com/manveertamber/pyserini/blob/master/docs/experiments-wiki-corpora.md) in our Pyserini IR library.


It's recommended to run the experiment in virtualenv (with Python-version=3.7).


End-to-end answer prediction

## Setup
```bash
pip install pyserini
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
git clone https://github.com/castorini/pygaggle.git 
cd pygaggle
pip install --editable .
pip install -r requirements.txt
```

Now download FiD

```bash
git clone https://github.com/facebookresearch/FiD.git
cd FiD
pip install -r requirements.txt
!pip install transformers==4.10.0
```

We make the FiD-large models available here: (You may use wget to download them)

[FiD-trivia model for wiki_our_100w_corpus](https://www.dropbox.com/s/i3az8xbgmqj6e3v/checkpoint_our_100w_trivia_large.tar.gz)  
[FiD-nq model for wiki_our_100w_corpus](https://www.dropbox.com/s/n9a01op57kyj54q/checkpoint_our_100w_nq_large.tar.gz)  
[FiD-trivia model for wiki_6_3_corpus](https://www.dropbox.com/s/8vlcrnkbtoa2mgh/checkpoint_6_3_trivia_large.tar.gz)  
[FiD-nq model for wiki_6_3_corpus](https://www.dropbox.com/s/0gk6ex2s7h3tr7t/checkpoint_6_3_nq_large.tar.gz)  
[FiD-trivia model for wiki_6_3_TL_corpus](https://www.dropbox.com/s/u1kv5zq4rvuzc5y/checkpoint_tables_6_3_trivia_large.tar.gz)  
[FiD-nq model for wiki_6_3_TL_corpus](https://www.dropbox.com/s/vv0bllh0o1u9s3a/checkpoint_tables_6_3_nq_large.tar.gz)  
[FiD-trivia model for wiki_8_4_corpus](https://www.dropbox.com/s/qdayolo1kjouq69/checkpoint_8_4_trivia_large.tar.gz)  
[FiD-nq model for wiki_8_4_corpus](https://www.dropbox.com/s/v88ja4tlbb7gw3k/checkpoint_8_4_nq_large.tar.gz)  
[FiD-trivia model for wiki_8_4_TL_corpus](https://www.dropbox.com/s/a8hc0sgvshj3rti/checkpoint_tables_8_4_trivia_large.tar.gz)  
[FiD-nq model for wiki_8_4_TL_corpus](https://www.dropbox.com/s/ae4fugw4yg02p7p/checkpoint_tables_8_4_nq_large.tar.gz)  

After downloading the model, don't forget to change the model name being used in pygaggle/qa/fid_reader.py

## Inference and Evaluation

Then we run the inference and evaluation in the TOP level directory of Pygaggle. The retrieval_file is either from retrieval from the 2nd iteration DPR model or from hybrid retrieval.

```bash
$ python -um pygaggle.run.evaluate_fid_reader \
    --model_name model_checkpoint \
    --retrieval_file retrieval_run.json \
    --output_file qa_run.json
```


## Results:

```bash
Trivia
| Setting              | Total (Dev) | Total |
|----------------------|-------------|-------|
| wiki_our_100w        | 70.4        | 70.4  |
| wiki_our_100w_hybrid |             | 72.4  |
| wiki_6_3             | 70.6        | 70.8  |
| wiki_6_3_hybrid      |             | 72.9  |
| wiki_8_4             | 70.2        | 70.3  |
| wiki_8_4_hybrid      |             | 72.5  |
| wiki_TL_6_3          | 71.8        | 71.7  |
| wiki_TL_6_3_hybrid   |             | 73.8  |
| wiki_TL_8_4          | 71.1        | 71.3  |
| wiki_TL_8_4_hybrid   |             | 73.5  |


NQ
| Setting              | Total (Dev) | Total |
|------------=---------|-------------|-------|
| wiki_our_100w        | 50.4        | 51.0  |
| wiki_our_100w_hybrid |             | 51.4  |
| wiki_6_3             | 51.0        | 52.4  |
| wiki_6_3_hybrid      |             | 52.9  |
| wiki_8_4             | 51.3        | 53.7  |
| wiki_8_4_hybrid      |             | 53.9  |
| wiki_TL_6_3          | 54.3        | 55.3  |
| wiki_TL_6_3_hybrid   |             | 56.4  |
| wiki_TL_8_4          | 54.0        | 55.2  |
| wiki_TL_8_4_hybrid   |             | 55.8  |


```


If you were able to replicate these results, please submit a PR adding to the replication log! Please mention in your PR if you find any difference!



