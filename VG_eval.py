from evaluation.sgg.VG.vg_eval import do_vg_evaluation
from evaluation.sgg.VG.config import _C as cfg
import json
import sys
import os.path as op
from data.mm_data.sgg_VG_dataset import SggVGDataset, VGDatasetReader
from data.mm_data.sg_raw import load_json
import logging




def evaluate_on_VG_seq(predictions, output_folder):
    img_dir = '/data/c/zhuowan/gqa/data/images/'
    base_dir = 'dataset/sgg_data/VG'
    roidb_file = f'{base_dir}/VG-SGG-with-attri.h5'
    dict_file = f'{base_dir}/VG-SGG-dicts-with-attri.json'
    image_file = f'{base_dir}/image_data.json'
    tgt_seq_len = 350
    split = 'train'
    # print("split: ", split)

    dataset = VGDatasetReader(
        split, 
        img_dir, 
        roidb_file, 
        dict_file, 
        image_file,
        bpe='gpt2',
        num_im=-1,
        num_val_im=5000,
        required_len=tgt_seq_len
    )

    iou_types = ['bbox', 'relations'] # 'bbox', 'relations'
    logger = logging.getLogger(__name__)


    do_vg_evaluation(
        cfg=cfg,
        dataset=dataset,
        predictions_raw=predictions,
        output_folder=output_folder,
        logger=logger,
        iou_types=iou_types
    )


if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     evaluate_on_VG_seq(sys.argv[1], sys.argv[2])
    # else:
    #     raise NotImplementedError
    predictions = load_json('results/sgg/VG/train_0305_PredCls_train_small_dataset_tiny_350_on_training_set.json')
    output_folder = 'results/sgg/VG/recall_evals/'
    evaluate_on_VG_seq(predictions, output_folder)