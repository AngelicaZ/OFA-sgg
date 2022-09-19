#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
bpe='gpt2'

data_dir=/data/c/zhuowan/gqa/data/sceneGraphs
data=${data_dir}/train_sceneGraphs.json,${data_dir}/val_sceneGraphs.json

checkpoint=checkpoint_best
paras=50_1e-4_ofa_tiny_350_v4
path=../../run_scripts/sgg/sgg_checkpoints/${paras}/${checkpoint}.pt
result_path=../../results/sgg

split='train'

log_dir=./sgg_logs_eval/GQA
mkdir -p $log_dir

log_file=${log_dir}/"trainset_"${paras}"_"${checkpoint}".log"

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe=${bpe} \
    --task=sgg \
    --batch-size=4 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=1000 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 > ${log_file} 2>&1 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False}"

# python GQA_eval.py ../../results/sgg/test_predict.json ${data_dir}/val_sceneGraphs.json