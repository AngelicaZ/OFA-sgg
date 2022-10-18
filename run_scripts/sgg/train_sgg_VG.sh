
# Number of GPUs per GPU worker
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=127.0.0.1 # 162.129.251.54

# The port for communication
export MASTER_PORT=1051
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0 

export CUDA_VISIBLE_DEVICES=4,5,6,7

dataset_choose='VG'


img_dir=/data/c/zhuowan/gqa/data/images/
base_dir=../../dataset/sgg_data/VG
data=${base_dir}/VG-SGG-with-attri.h5
dict_file=${base_dir}/VG-SGG-dicts-with-attri.json
image_file=${base_dir}/image_data.json

log_dir=./sgg_logs/VG
save_dir=./sgg_checkpoints/VG
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
bpe='gpt2'
user_dir=../../ofa_module

task=sgg
arch=ofa_tiny
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
report_accuracy=True
batch_size=4
warmup_ratio=0.06
total_num_update=50000
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_object_length=100
max_tgt_length=1000
num_bins=1000


patch_image_size=512

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

tgt_seq_len=350


#           --roidb-file=${roidb_file} \
# --warmup-ratio=${warmup_ratio} \
# --lr-scheduler=polynomial_decay \
# --total-num-update=${total_num_update} \

for max_epoch in 10; do
  echo "max_epoch "${max_epoch}
  for lr in 1e-4; do
    echo "lr "${lr}
    for patch_image_size in 512; do
      echo "arch "${arch}
      echo "target_seq_len "${tgt_seq_len}

      log_file=${log_dir}/"VG_1017_l2loss_"${max_epoch}"_"${lr}"_"${arch}"_"${tgt_seq_len}".log"
      save_path=${save_dir}/"VG_1017_l2loss_"${max_epoch}"_"${lr}"_"${arch}"_"${tgt_seq_len}
      mkdir -p $save_path

      python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
          ${data} \
          --bpe-dir=${bpe_dir} \
          --bpe=${bpe} \
          --user-dir=${user_dir} \
          --dataset-choose=${dataset_choose} \
          --tgt-seq-len=${tgt_seq_len} \
          --dict-file=${dict_file} \
          --image-file=${image_file} \
          --img-dir=${img_dir} \
          --restore-file=${restore_file} \
          --reset-optimizer --reset-dataloader --reset-meters \
          --save-dir=${save_path} \
          --task=${task} \
          --arch=${arch} \
          --criterion=${criterion} \
          --label-smoothing=${label_smoothing} \
          --report-accuracy \
          --batch-size=${batch_size} \
          --update-freq=${update_freq} \
          --encoder-normalize-before \
          --decoder-normalize-before \
          --share-decoder-input-output-embed \
          --share-all-embeddings \
          --layernorm-embedding \
          --patch-layernorm-embedding \
          --code-layernorm-embedding \
          --resnet-drop-path-rate=${resnet_drop_path_rate} \
          --encoder-drop-path-rate=${encoder_drop_path_rate} \
          --decoder-drop-path-rate=${decoder_drop_path_rate} \
          --dropout=${dropout} \
          --attention-dropout=${attention_dropout} \
          --weight-decay=0.01 \
          --optimizer=adam \
          --adam-betas="(0.9,0.999)" \
          --adam-eps=1e-08 \
          --clip-norm=1.0 \
          --lr-scheduler=polynomial_decay \
          --lr=${lr} \
          --warmup-ratio=${warmup_ratio} \
          --max-epoch=${max_epoch} \
          --log-format=simple \
          --log-interval=10 \
          --fixed-validation-seed=7 \
          --keep-best-checkpoints=1 \
          --keep-last-epochs=15 \
          --save-interval=1 --validate-interval=5 \
          --save-interval-updates=500 --validate-interval-updates=500 \
          --eval-bleu \
          --eval-args='{"beam":5,"max_len":1000,"no_repeat_ngram_size":3}' \
          --best-checkpoint-metric=bleu --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --num-workers=0 > ${log_file} 2>&1
    done
  done
done
