# PyGaggle: Closed Book Question Answering Baselines

This page contains instructions for running the Closed Book Question Answering.

First install Pygaggle

```
git clone --recursive https://github.com/castorini/pygaggle.git
```

Then

```
pip install pygaggle/
```

## Models

+ How Much Knowledge Can You Pack Into the Parameters of a Language Model? [(Adam Roberts, Colin Raffel, Noam Shazeer, 2020)](https://arxiv.org/pdf/2002.08910.pdf)

## Natural Questions (NQ)

### Using t5-large-ssm-nq

```bash
python -um pygaggle.run.evaluate_closed_book_question_answering --data nq \
                                                                --model-name google/t5-large-ssm-nq \
                                                                --device cuda:0
```

The following output will be visible after it has finishes

```
Exact Match Accuracy: 29.861495844875346
```

### Using t5-xl-ssm-nq

Note: This model is slightly over 10GB in size. If you wish to run this on a GPU, please choose one with > 10GB memory.
```bash
python -um pygaggle.run.evaluate_closed_book_question_answering --data nq \
                                                                --model-name google/t5-xl-ssm-nq \
                                                                --device cuda:0
```

The following output will be visible after it has finishes

```
Exact Match Accuracy: 34.044321329639885
```

### Using t5-3b-ssm-nq

Note: This model is slightly over 10GB in size. If you wish to run this on a GPU, please choose one with > 10GB memory.
```bash
python -um pygaggle.run.evaluate_closed_book_question_answering --data nq \
                                                                --model-name google/t5-3b-ssm-nq \
                                                                --device cuda:0
```

The following output will be visible after it has finishes

```
Exact Match Accuracy: 31.191135734072024
```

If you were able to replicate these results, please submit a PR adding to the replication log!
Please mention in your PR if you find any difference!


## Replication Log
