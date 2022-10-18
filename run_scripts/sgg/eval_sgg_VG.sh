
export MASTER_PORT=8087
# 8087, 4081, 3053, 3054, 3055
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export NCCL_P2P_LEVEL=NVL

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
bpe='gpt2'

# img_dir=/data/c/zhuowan/gqa/data/images/
base_dir=../../dataset/sgg_data/VG
data=${base_dir}/VG-SGG-with-attri.h5
# dict_file=${base_dir}/VG-SGG-dicts-with-attri.json
# image_file=${base_dir}/image_data.json

checkpoint=checkpoint_best
paras=VG_1016_nobbox_10_1e-4_ofa_tiny_350
path=../../run_scripts/sgg/sgg_checkpoints/VG/${paras}/${checkpoint}.pt
result_path=../../results/sgg/VG
# selected_cols=1,4,2
split='test'

log_dir=./sgg_logs_eval/VG
mkdir -p $log_dir

log_file=${log_dir}/${paras}"_"${checkpoint}".log"

python3 -m torch.distributed.launch  --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
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

# python VG_eval.py ../../results/sgg/test_1007l2_predict.json_predict.json ../../dataset/sgg_data/VG/test_caption_coco_format.json