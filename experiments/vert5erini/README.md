## Start a VM with TPU on Google Cloud

Define environment variables.
```
export PROJECT_NAME=<gcloud project name>
export PROJECT_ID=<gcloud project id>
export INSTANCE_NAME=<name of vm to create>
export TPU_NAME=<name of tpu to create>
```

Create the VM.
```
gcloud beta compute --project=${PROJECT_NAME} instances create ${INSTANCE_NAME} --zone=europe-west4-a --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=${PROJECT_ID}-compute@developer.gserviceaccount.com  --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-10-buster-v20201112 --image-project=debian-cloud --boot-disk-size=25GB --boot-disk-type=pd-standard --boot-disk-device-name=${INSTANCE_NAME} --reservation-affinity=any
```

It is possible that the `image` and `machine-type` provided here are dated so feel free to update them to whichever fits your needs.
After the VM created, we can `ssh` to the machine.  
Make sure to initialize `PROJECT_NAME` and `TPU_NAME` from within the machine too.
Then create a TPU.

```
curl -O https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu
./ctpu up --name=${TPU_NAME} --project=${PROJECT_NAME} --zone=europe-west4-a --tpu-size=v3-8 --tpu-only --noconf
```

## Setup environment on VM

Install required tools including [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
```
sudo apt-get update
sudo apt-get install git gcc screen make openjdk-11-jdk-headless --yes
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Then create a Python virtual environment for the experiments and install dependencies.
```
conda init
conda create --y --name py36 python=3.6
conda activate py36
conda install -c conda-forge httptools jsonnet --yes
pip install tensorflow==2.3 tensorflow-text==2.3 t5[gcp]==0.6.4 pyserini jsonlines
git clone https://github.com/castorini/mesh.git
pip install --editable mesh
```

## Download Dataset

Please download the dataset by follow the guidance in original [SciFact](https://github.com/allenai/scifact) repo

## Abstract Retrieval

Please download the prebuild SciFact index [here](https://www.dropbox.com/s/khzy7r0pk124cpx/scifact_index.zip)

```
python bm25_retrieval.py --index scifact_index \
                         --claims claims_dev.jsonl \
                         --topk 20 \
                         --results bm25_retrieval_top20_dev.jsonl
```

## Abstract Reranking (AR)

1. Create input file for T5 inference
```
python prepare_ar_input.py --corpus corpus.jsonl \
                           --claims claims_dev.jsonl \
                           --retrieval bm25_retrieval_top20_dev.jsonl \
                           --t5_input_ids ar_inference_dev_ids.txt \
                           --t5_input ar_inference_dev.txt
```

2. Run T5 inference for abstract rerank
```
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT_NAME}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="gs://castorini/monot5/experiments/3B" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="infer_checkpoint_step = 1010000" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="input_filename = 'ar_inference_dev.txt'" \
  --gin_param="output_filename = 'ar_inference_dev_monot5-3b_output.txt'" \
  --gin_param="tokens_per_batch = 65536" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1"
```

3. Convert T5 output to abstract retrieval/rerank result file (in `jsonl` format)
```
python create_ar_result.py --t5_output_ids ar_inference_dev_ids.txt \
                           --t5_output ar_inference_dev_monot5-3b_output.txt-1010000 \
                           --topk 3 \
                           --results ar_dev.jsonl
```

## Sentence Selection (SS)

1. Prepare T5 input for sentence selection from the abstract retrieval result
```
python prepare_ss_input.py --corpus corpus.jsonl \
                           --claims claims_dev.jsonl \
                           --retrieval ar_dev.jsonl \
                           --t5_input_ids ss_inference_dev_ids.txt \
                           --t5_input ss_inference_dev.txt
```

2. Run T5 inference for sentence selection
```
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project=${PROJECT_NAME} \
  --tpu_zone="europe-west4-a" \
  --model_dir="gs://pongo-bucket/xueguang/vert5-repl/ss-train" \
  --gin_file="operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="infer_checkpoint_step = 1012500" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="input_filename = 'ss_inference_dev.txt'" \
  --gin_param="output_filename = 'ss_inference_dev_monot5-3b_output.txt'" \
  --gin_param="tokens_per_batch = 65536" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1"
```

3. Convert T5 output to sentence selection result file (in `jsonl` format)
```
python create_ss_result.py --claims claims_dev.jsonl \
                           --t5_output_ids ss_inference_dev_ids.txt \
                           --t5_output ss_inference_dev_monot5-3b_output.txt-1012500 \
                           --results ss_dev.jsonl
```

## Label Prediction (LP)

1. Prepare T5 input for label prediction from sentence selection result
```
python prepare_lp_input.py --corpus corpus.jsonl \
                           --claims claims_dev.jsonl \
                           --sentence_selection ss_dev.jsonl \
                           --t5_input_ids lp_inference_dev_ids.txt \
                           --t5_input lp_inference_dev.txt
```

2. Run T5 inference for label prediction
```
t5_mesh_transformer \
  --tpu="${TPU_NAME}" \
  --gcp_project=${PROJECT_NAME} \
  --tpu_zone="europe-west4-a" \
  --model_dir="gs://pongo-bucket/xueguang/vert5-repl/lp-train" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="infer_checkpoint_step = 1000600" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2" \
  --gin_param="input_filename = 'lp_inference_dev.txt'" \
  --gin_param="output_filename = 'lp_inference_dev_monot5-3b_output.txt'" \
  --gin_param="tokens_per_batch = 65536" \
  --gin_param="Bitransformer.decode.beam_size = 1" \
  --gin_param="Bitransformer.decode.temperature = 0.0" \
  --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1"
```

3. Convert T5 output to label prediction result file (in `jsonl` format)
```
python create_lp_result.py --t5_output_ids lp_inference_dev_ids.txt \
                           --t5_output lp_inference_dev_monot5-3b_output.txt-1000600 \
                           --claims  claims_dev.jsonl \
                           --results lp_dev.jsonl
```

## Evaluate

1. Create full pipeline evaluation file
```
python create_full_pipe_eval.py --ar_result ar_dev.jsonl \
                                --ss_result ss_dev.jsonl \
                                --lp_result lp_dev.jsonl \
                                --lp_eval full_pipeline_eval.jsonl
```

2. Evaluate full pipeline
Please download the evaluate folder from original [SciFact](https://github.com/allenai/scifact) repo. 

```
python evaluate/pipeline.py --gold claims_dev.jsonl \
                        --corpus corpus.jsonl \
                        --prediction full_pipeline_eval.jsonl
```
We can expect to see the following results for the full pipeline evaluation of the development set:

|          | sentence_selection | sentence_label | abstract_label_only | abstract_rationalized |
|---|---|---|---|---|
|precision |          0.659164  |   0.633441     |        0.671717        |       0.641414 |
|recall    |          0.560109  |   0.538251     |        0.636364        |       0.607656 |
|f1         |         0.605613  |   0.581979     |        0.653563        |       0.624079 |


## Train
### Sentence Selection Training
Generate Sentence Selection training data
```bash
python prepare_ss_train_input.py --corpus corpus.jsonl \
                                 --claims claims_train.jsonl \
                                 --t5_input ss_train.txt \
                                 --balanced
shuf ss_train.txt > ss_train_shuf.txt
```

Train
```bash
export MODEL_NAME=3B
export GS_FOLDER=<gs folder to save model>
export MODEL_INIT_CKPT=1010000
echo "model_checkpoint_path: \"model.ckpt-${MODEL_INIT_CKPT}\"" > checkpoint
gsutil cp checkpoint ${GS_FOLDER}
gsutil cp gs://neuralresearcher_data/doc2query/experiments/363/model.ckpt-${MODEL_INIT_CKPT}* ${GS_FOLDER}
gsutil cp ss_train_shuf.txt ${GS_FOLDER}/ss_train_shuf.txt

t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT_NAME}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="${GS_FOLDER}" \
  --gin_param="init_checkpoint = 'gs://neuralresearcher_data/doc2query/experiments/363/model.ckpt-${MODEL_INIT_CKPT}'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = '${GS_FOLDER}/ss_train_shuf.txt'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1012500" \
  --gin_param="run.save_checkpoints_steps = 200" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)"
```

### Label Prediction Training
Generate Label Prediction training data
```bash
python prepare_lp_train_input.py --corpus corpus.jsonl \
                                 --claims claims_train.jsonl \
                                 --t5_input lp_train.txt
shuf lp_train.txt > lp_train_shuf.txt
```

Train
```bash
export MODEL_NAME=3B
export GS_FOLDER=<gs folder to save model>
export MODEL_INIT_CKPT=1000000
echo "model_checkpoint_path: \"model.ckpt-${MODEL_INIT_CKPT}\"" > checkpoint
gsutil cp checkpoint ${GS_FOLDER}
gsutil cp gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}* ${GS_FOLDER}
gsutil cp lp_train_shuf.txt ${GS_FOLDER}/lp_train_shuf.txt

t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT_NAME}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="${GS_FOLDER}" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = '${GS_FOLDER}/lp_train_shuf.txt'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1000600" \
  --gin_param="run.save_checkpoints_steps = 200" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)"
```

### Checkpoints:
- sentence selection: `gs://pongo-bucket/xueguang/vert5-repl/ss-train`
- label prediction: `gs://pongo-bucket/xueguang/vert5-repl/lp-train`
- sentence selection (with dev): `gs://pongo-bucket/xueguang/vert5erini/train/monot5-3B-LP-balanced-dev`
- label prediction (with dev): `gs://pongo-bucket/xueguang/vert5erini/train/monot5-3B-SS-balanced-dev`

### Evaluate Each Step
3. Evaluate abstract retrieval
```
python evaluate/abstract_retrieval.py --dataset claims_dev.jsonl \
                                      --abstract-retrieval ar_dev.jsonl
```
We can expect to see the following results for the retrieval stage of the pipeline
```
Hit one: 0.9567
Hit all: 0.9367
```

4. Evaluate sentence selection
```
python evaluate/rationale_selection.py --corpus corpus.jsonl \
                                       --dataset claims_dev.jsonl \
                                       --rationale-selection ss_dev.jsonl
```
We can expect to see the following results for the sentence selection stage of the pipeline
```
{'precision': 0.6404833836858006, 'recall': 0.5792349726775956, 'f1': 0.6083213773314203}
```

5. Evaluate label prediction
```
python evaluate/label_prediction.py --corpus corpus.jsonl \
                                    --dataset claims_dev.jsonl \
                                    --label-prediction lp_dev.jsonl
```
We can expect to see the following results for the label prediction stage of the pipeline
```
Accuracy           0.6385
Macro F1:          0.5002
Macro F1 w/o NEI:  0.7502
                   [C      N      S     ]
F1:                [0.735  0.     0.7654]
Precision:         [0.6232 0.     0.6596]
Recall:            [0.8958 0.     0.9118]
Confusion Matrix:
[[43  1  4]
 [19  0 44]
 [ 7  2 93]]
```

### Evaluate With Oracle Setting
To evaluate with Oracle setting, just replace the input of each inference step by oracle files from SciFact.

E.g. If you want to evaluate Sentence Selection in oracle setting, when you run `prepare_ss_input.py`, just replace the `ar_dev.jsonl` by oracle retrieval file that identical to SciFact.


## Replication Log




