# Setup a VM on Google Cloud

## Multi-Stage Relevance Ranking

Define environment variables.
```bash
export PROJECT_NAME=<gcloud project name>
export PROJECT_ID=<gcloud project id>
export INSTANCE_NAME=<name of vm to create>
export TPU_NAME=<name of tpu to create>
```

Create the VM.
```bash
gcloud beta compute --project=${PROJECT_NAME} instances create ${INSTANCE_NAME} --zone=europe-west4-a --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=${PROJECT_ID}-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-10-buster-v20210721 --image-project=debian-cloud --boot-disk-size=25GB --boot-disk-type=pd-balanced --boot-disk-device-name=${INSTANCE_NAME} --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
gcloud beta services identity create --service tpu.googleapis.com --project ${PROJECT_NAME}
```

It is possible that the `image` and `machine-type` provided here are dated so feel free to update them to whichever fits your needs.
After the VM created, we can `ssh` to the machine.  
Make sure to initialize `PROJECT_NAME` and `TPU_NAME` from within the machine too.


Setup environment on VM
```bash
export MODEL_NAME=${3:-3B}
export TASK=${2:-trec-misinfo}
export EXPNO=${1:-1}
export BUCKET=<bucket name>
export PROJECT_NAME=<gcloud project name>
export TPU_NAME=<name of tpu to create>

curl -O https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu
sudo apt-get update
sudo apt-get install git gcc screen --yes
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```

Set up conda virtual environment and create tpu
```bash
conda init
conda create --y --name vera python=3.7
conda activate vera
conda install -c conda-forge httptools jsonnet --yes
pip install tensorflow==2.5 tensorflow-text 
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
cd text-to-text-transfer-transformer && git checkout ca1c0627f338927ac753159cb7a1f6caaf2ae19b && pip install --editable . && cd ..
pip install cloud-tpu-client
git clone https://github.com/castorini/mesh.git
pip install --editable mesh/
./ctpu up --name=${TPU_NAME} --project=${PROJECT_NAME} --zone=europe-west4-a --tpu-size=v3-8 --tf-version=2.5.0 --tpu-only --noconf
```

### MonoT5 (base and NL)

Create monoT5 input files for inference
```bash

export TAG=ref.hm20
export IN_DIR=<input data directory>

python3 experiments/vera/create_hm_qrels_monot5_input.py --stride 3 --length 6 \
	--queries topics/reformulated-topics-misinfo-2020.queries.tsv \
	--qrels  runs/run.hm20-bm25.trec \
	--index collections/trec-misinfo/warc_index \
	--t5_input ${IN_DIR}/query_doc_pairs.${TAG}.txt \
       	--t5_input_ids ${IN_DIR}/query_doc_pair_ids_${TAG}.tsv
```

Then send the t5 input file to google bucket with `gsutil cp input_file_dir google_bucket_dir`

Run inference with monot5
```bash

export DATA_DIR=$BUCKET/$TASK/data/${EXPNO} 
export MODEL_DIR=gs://castorini/med-monot5/experiments/3B 
export TAG=ref.hm20
for ITER in {000..001}; do 
        echo "Running iter: $ITER" >> out.log_eval_exp 
        nohup t5_mesh_transformer \
                --tpu="${TPU_NAME}" \
                --gcp_project="${PROJECT_NAME}" \
                --tpu_zone="europe-west4-a" \
                --model_dir="${MODEL_DIR}" \
                --gin_file="gs://t5-data/pretrained_models/${MODEL_NAME}/operative_config.gin" \
                --gin_file="infer.gin" \
                --gin_file="beam_search.gin" \
                --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
                --gin_param="infer_checkpoint_step = 1011000" \
                --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
                --gin_param="Bitransformer.decode.max_decode_length = 2" \
                --gin_param="input_filename = '${DATA_DIR}/query_doc_pairs.${TAG}.txt${ITER}'" \
                --gin_param="output_filename = '${DATA_DIR}/${MODEL_NAME}_query_doc_pair_scores.${TAG}.txt${ITER}
'" \
                --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
                --gin_param="Bitransformer.decode.beam_size = 1" \
                --gin_param="Bitransformer.decode.temperature = 0.0" \
                --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
                >> out.log_eval_exp 2>&1 
done &
tail -100f out.log_eval_exp
```

Then, download the monot5 score file and convert to trec format
```bash

export DATA_DIR=$BUCKET/$TASK/data/${EXPNO} 
export TAG=ref.hm20

python experiments/vera/convert_monot5_output_to_marco_run.py --predictions ${DATA_DIR}/3B_query_doc_pair_scores.${TAG}.txt-1011000 --query_run_ids  ${DATA_DIR}/query_doc_pair_ids_${TAG}.tsv --output ${DATA_DIR}/run.monot5_3B.${TAG}.trec --output_seg ${DATA_DIR}/seg.run.monot5_3B.${TAG}.trec

sed -E -i "s/<urn:uuid:(.*)>/\1/" ${DATA_DIR}/run.monot5_3B.${TAG}.trec
```

And we can finally run evaluation on the monot5 run files.

Note, we can choose to either have ref or normal queries.

### DuoT5 (base and NL)


Create input files for duot5 inference on TPU
```bash

python3 trec-hm-20/21/create_hm_qrels_duot5_input.py --stride 3 --length 6 \
	--queries topics/reformulated-topics-misinfo-2020.queries.tsv \
	--run runs/seg.run.monot5_3B.${TAG}.trec \
	--index collections/trec-misinfo/warc_index \
	--t5_input ${DATA_DIR}/query_doc_triples.${TAG}.txt \
       	--t5_input_ids ${DATA_DIR}/query_doc_triple_ids_${TAG}.tsv
```

Run duot5 inference on VM
```bash
export GS_FOLDER=$BUCKET/$URA/$TASK/data/${EXPNO}
export OUT_FOLDER=$BUCKET/$URA/$TASK/data/${EXPNO} 
export MODEL_DIR=gs://castorini/med-duot5/experiments/3B 
export TAG=ref.hm20
nohup t5_mesh_transformer \
	--tpu="${TPU_NAME}" \
	--gcp_project="${PROJECT_NAME}" \
	--tpu_zone="europe-west4-a" \
	--model_dir="${MODEL_DIR}" \
	--gin_file="gs://t5-data/pretrained_models/${MODEL_NAME}/operative_config.gin" \
	--gin_file="infer.gin" \
	--gin_file="beam_search.gin" \
	--gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
	--gin_param="infer_checkpoint_step = 1011000" \
	--gin_param="utils.run.sequence_length = {'inputs': 1024, 'targets': 2}" \
	--gin_param="Bitransformer.decode.max_decode_length = 2" \
	--gin_param="input_filename = '${DATA_DIR}/query_doc_triples.${TAG}.txt'" \
	--gin_param="output_filename = '${DATA_DIR}/${MODEL_NAME}_query_doc_triple_scores.${TAG}.txt'" \
	--gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
	--gin_param="Bitransformer.decode.beam_size = 1" \
	--gin_param="Bitransformer.decode.temperature = 0.0" \
	--gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
	>> out.log_eval_exp 2>&1 &
tail -100f out.log_eval_exp
```
Download the duot5 score file and convert to trec format
```bash

python trec-hm-20/21/convert_duot5_output_to_trec.py --predictions ${DATA_DIR}/3B_query_doc_triple_scores.${TAG}.txt-1011000 --query_run_ids ${DATA_DIR}/query_doc_triple_ids_${TAG}.tsv --output ${DATA_DIR}/run.duot5_3B.${TAG}.trec

sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/run.duot5_3B.${TAG}.trec
```

Combine duot5 run file with monot5 run file using two weighting schemes described in the paper.
```bash

python3 trec-hm-20/21/merge_mono_duo_d1.py --run_mono ${OUT_DIR}/runs/run.monot5_3B.${TAG}.trec --run_duo ${DATA_DIR}/run.duot5_3B.${TAG}.trec --output_run ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d1.${TAG}.trec

python3 trec-hm-20/21/merge_mono_duo_d2.py --run_mono ${OUT_DIR}/runs/run.monot5_3B.${TAG}.trec --run_duo ${DATA_DIR}/run.duot5_3B.${TAG}.trec --output_run ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d2.${TAG}.trec;
```
Finally, we run evaluations with combined duo run file.


## Vera

Create input file for vera 
```bash

export TAG=ref.hm20
python3 experiments/vera/create_hm_qrels_vera_input.py --stride 3 --length 6 \
	--queries topics/topics-misinfo-2020.queries.tsv \
	--run ${IN_DIR}/seg.run.monot5_3B.${TAG}.trec \
	--index collections/trec-misinfo/warc_index \
	--t5_input ${IN_DIR}/query_doc_vera.${TAG}.txt \
       	--t5_input_ids ${IN_DIR}/query_doc_vera_ids_${TAG}.tsv
```

Run inference with vera on VM
```bash
export MODEL_DIR=gs://castorini/vera/experiments/3B 
export TAG=ref.hm20
echo "Running iter: $ITER" >> out.log_eval_exp 
nohup t5_mesh_transformer \
	--tpu="${TPU_NAME}" \
	--gcp_project="${PROJECT_NAME}" \
	--tpu_zone="europe-west4-a" \
	--model_dir="${MODEL_DIR}" \
	--gin_file="gs://t5-data/pretrained_models/${MODEL_NAME}/operative_config.gin" \
	--gin_file="infer.gin" \
	--gin_file="beam_search.gin" \
	--gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
	--gin_param="infer_checkpoint_step = 1000500" \
	--gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
	--gin_param="Bitransformer.decode.max_decode_length = 2" \
	--gin_param="input_filename = '${DATA_DIR}/query_doc_vera.${TAG}.txt'" \
	--gin_param="output_filename = '${DATA_DIR}/${MODEL_NAME}_query_doc_vera_scores.${TAG}.txt'" \
	--gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
	--gin_param="Bitransformer.decode.beam_size = 1" \
	--gin_param="Bitransformer.decode.temperature = 0.0" \
	--gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
	>> out.log_eval_exp 2>&1 &
tail -100f out.log_eval_exp
```
Download score file for vera and convert to trec run file
```bash

python trec-hm-20/21/convert_vera_output_to_trec.py --input_topic /store/share/trec-misinfo/topics/misinfo-2020-topics.xml --year 2020 --predictions ${DATA_DIR}/3B_query_doc_vera_scores.${TAG}.txt-1000500 --query_run_ids ${DATA_DIR}/query_doc_vera_ids_${TAG}.tsv --output ${OUT_DIR}/run.vera_3B.${TAG}.trec

sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/run.vera_3B.${TAG}.trec
```
Run evaluation with vera run file without linear combinations

Combine the vera run file with the mono/duo run file
Here, we include a few linear combinations constants that we reported in Vera:
```bash

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.95 --output ${OUT_DIR}/runs/run.mono.vera.0.95.${TAG}.trec

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d1.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.95 --output ${OUT_DIR}/runs/run.duo1.vera.0.95.${TAG}.trec

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d2.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.75 --output ${OUT_DIR}/runs/run.duo2.vera.0.75.${TAG}.trec

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.5 --output ${OUT_DIR}/runs/run.mono.vera.0.5.${TAG}.trec

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d1.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.5 --output ${OUT_DIR}/runs/run.duo1.vera.0.5.${TAG}.trec

python3 trec-hm-20/21/combine_runs.py --run_a ${OUT_DIR}/runs/run.monot5_3B.duot5_3B.d2.${TAG}.trec --run_b ${DATA_DIR}/runs/run.vera_3B.ref.hm20.trec --alpha=0.5 --output ${OUT_DIR}/runs/run.duo2.vera.0.5.${TAG}.trec

sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/runs/run.mono.vera.0.95.${TAG}.trec;
sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/runs/run.duo1.vera.0.95.${TAG}.trec;
sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/runs/run.duo2.vera.0.75.${TAG}.trec;
sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/runs/run.mono.vera.0.5.${TAG}.trec;
sed -E -i "s/<urn:uuid:(.*)>/\1/" ${OUT_DIR}/runs/run.duo1.vera.0.5.${TAG}.trec;
```

Run evaluation with these combined files
