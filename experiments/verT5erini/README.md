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
  --model_dir="gs://castorini/med-monot5/experiments/3B" \
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
  --model_dir="gs://pongo-bucket/scifact/experiments/6" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
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
  --model_dir="gs://pongo-bucket/scifact/experiments/9" \
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

## Result

We can expect to see following results for the full pipeline evaluation of the development set

|          | sentence_selection | sentence_label | abstract_label_only | abstract_rationalized |
|---|---|---|---|---|
|precision |          0.644172  |   0.604294     |        0.650718        |       0.617225 |
|recall    |          0.573770  |   0.538251     |        0.650718        |       0.617225 |
|f1         |         0.606936  |   0.569364     |        0.650718        |       0.617225 |

## Replication Log





