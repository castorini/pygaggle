# PyGaggle: Fusion-in-Decoder (FiD) Baselines

This page contains instructions for running the FiD tools.

It's recommended to run the experiment in virtualenv (with Python-version=3.7).

## Natural Questions (NQ)
End-to-end answer prediction using **Hybrid Retrieval**

Run nvidia-smi to make sure gpu is set up.
```bash
pip install pyserini
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
git clone https://github.com/castorini/pygaggle.git 
cd pygaggle
pip install --editable .
pip install -r requirements.txt
```

Now download the model accordingly, and the model name can be found in FiD/get-model.sh. (i.e. nq_reader_base)

```bash
git clone https://github.com/facebookresearch/FiD.git
cd FiD
bash get-model.sh -m nq_reader_base
pip install -r requirements.txt
!pip install transformers==4.10.0
```

After downloading the model, don't forget to change the model name being used in pygaggle/qa/fid_reader.py

Then we run the inference and evaluation in the TOP level directory of Pygaggle:
```bash
$ python -um pygaggle.run.evaluate_fid_ranker --model-name nq_reader_large \
                                            --retrieval-file data/run.nq-test.dkrr.gar.hybrid.json \
                                            --output-file data/fid_large.dkrr.gar.hybrid.out
```

Finally, we can analyze the result file using pygaggle/scripts/evaluate_qa_overlap_em.py.

```bash
$ python scripts/evaluate_qa_overlap_em.py --predictions reader_output.nq_test.fid_base.json \
                                 --dataset_name naturalquestions
```

With result (with bm25 and tqa & large model):

```bash
reader_output.bm25_nq_test.fid_large.json
--------------------------------------------------
Label       : total
N examples  :  3610
Exact Match :  54.930747922437675
--------------------------------------------------
Label       : question_overlap
N examples  :  324
Exact Match :  77.46913580246914
--------------------------------------------------
Label       : no_question_overlap
N examples  :  672
Exact Match :  43.601190476190474
--------------------------------------------------
Label       : answer_overlap
N examples  :  2297
Exact Match :  65.82498911623857
--------------------------------------------------
Label       : no_answer_overlap
N examples  :  1313
Exact Match :  35.872048743335874
--------------------------------------------------
Label       : answer_overlap_only
N examples  :  315
Exact Match :  48.888888888888886
--------------------------------------------------
Label       : no_overlap
N examples  :  357
Exact Match :  38.93557422969187
```


## Trivia QA (TQA)
End-to-end answer prediction using **Hybrid Retrieval**

Run nvidia-smi to make sure gpu is set up.
```bash
pip install pyserini
export JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
git clone https://github.com/castorini/pygaggle.git 
cd pygaggle
pip install --editable .
pip install -r requirements.txt
```

## download the tqa dataset (link:)

Now download the model accordingly, and the model name can be found in FiD/get-model.sh. (i.e. tqa_reader_base)

```bash
git clone https://github.com/facebookresearch/FiD.git
cd FiD
bash get-model.sh -m tqa_reader_base
pip install -r requirements.txt
!pip install transformers==4.10.0
```

After downloading the model, don't forget to change the model name being used in pygaggle/qa/fid_reader.py

Then we run the inference and evaluation in the TOP level directory of Pygaggle:
```bash
$ python -um pygaggle.run.evaluate_fid_ranker --task wikipedia --retriever score --reader fid \
            --settings dpr --retrieval-file data/run.encoded.dkrr.test.json --topk-em 100
```


Finally, we can analyze the result file using pygaggle/scripts/evaluate_qa_overlap_em.py.

```bash
$ python scripts/evaluate_qa_overlap_em.py --predictions reader_output.bm25_tqa_test.fid_large.json \
                                 --dataset_name triviaqa
```

With result (with bm25 and tqa & large model):

```bash
reader_output.bm25_tqa_test.fid_large.json
--------------------------------------------------
Label       : total
N examples  :  11313
Exact Match :  72.94263236984001
--------------------------------------------------
Label       : question_overlap
N examples  :  336
Exact Match :  93.75
--------------------------------------------------
Label       : no_question_overlap
N examples  :  665
Exact Match :  67.06766917293233
--------------------------------------------------
Label       : answer_overlap
N examples  :  8112
Exact Match :  83.01282051282051
--------------------------------------------------
Label       : no_answer_overlap
N examples  :  3201
Exact Match :  47.422680412371136
--------------------------------------------------
Label       : answer_overlap_only
N examples  :  411
Exact Match :  74.69586374695864
--------------------------------------------------
Label       : no_overlap
N examples  :  254
Exact Match :  54.724409448818896
```


# GPU and approximate time
Using Tesla P40 on datasci server, 
NQ base: 1~2 hours
NQ large: 4~5 hours
TQA base: ~5 hours
TQA large: ~15 hours

If you were able to replicate these results, please submit a PR adding to the replication log! Please mention in your PR if you find any difference!



