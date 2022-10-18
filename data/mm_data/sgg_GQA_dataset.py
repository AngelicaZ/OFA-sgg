
import json
import pdb
from posixpath import split
import time
import numpy as np
from data.mm_data.sg_raw import GQASceneDataset, load_json, np_load
import h5py
import logging
from typing import Iterable, Iterator, Callable, Dict
import torch
from torch.utils.data import Dataset
from data.ofa_dataset import OFADataset
import cv2
from data import data_utils
import utils.transforms as T
from torchvision import transforms
from PIL import Image, ImageFile


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}
    
    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )
    
    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()
    
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }
    # print("target in collate func shape: ", target.shape)

    return batch


class SggGQADataset(OFADataset):
    '''
    Dataset reader for scene graph data
    '''
    def __init__(self,
                 split,
                 dataset,
                 bpe,
                 src_dict,
                 tgt_dict,
                 max_tgt_length=30,
                 patch_image_size=512,
                 imagenet_default_mean_and_std=False,
                 max_image_size=512
                 ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.dataset = dataset
        self.img_dir = '/data/c/zhuowan/gqa/data/images/'
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
    
        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        
        # count how many images are loaded
        self.count = 0

    def __getitem__(self, idx):
        self.count += 1
        tgt_seq, imageid, src_text = self.dataset[idx]
        # src_features = torch.Tensor(src_features)
        # tgt_seq_ids = torch.Tensor(tgt_seq_ids)

        img_path = self.img_dir + imageid + ".jpg"
        img = cv2.imread(img_path)
        im_pil = Image.fromarray(img)
        patch_image = self.patch_resize_transform(im_pil)
        # img_resized = cv2.resize(img, (384, 512))
        # img = torch.Tensor(img_resized)

        '''tgt seq encoding approach 1'''
        # target seq len: [192]
        tgt_caption = '&&'.join(tgt_seq)
        tgt_item = self.encode_text(" {}".format(tgt_caption))  # NOTICE: removed space (recovered)
        # target raw original len: [406]

        '''tgt seq encoding approach 2'''
        # tgt_seq = ' '.join(tgt_seq)
        # tgt_caption = self.pre_caption(tgt_seq)
        # tgt_item = self.encode_text(tgt_caption)

        # print("tgt_caption: ", tgt_caption)
        # print("tgt_item: ", tgt_item)
        
        patch_mask = torch.tensor([True])

        src_caption = self.pre_caption(src_text)
        src_item = self.encode_text(src_caption)
        # print("src_caption: ", src_caption)
        # print("src_item: ", src_item)
        # pdb.set_trace()

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        # print("target original shape: ", target_item.shape)
        
        
        example = {
            "id": imageid,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example


    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
    


class GQADatasetReader(Dataset):
    '''
    Dataset reader for GQA scene graph data
    '''
    def __init__(self,
                 scenegraphs_json,
                 img_dir,
                 sgg_features_json_file,
                 sgg_features_h5_file,
                 embedding_json_path,
                 vocab_json_path,
                 new_vocab_json_path,
                 tgt_seq_len,
                 num_bins
                 ):
        super().__init__()
        self.embedding_json = np.load(embedding_json_path)
        self.sg_features_json = load_json(sgg_features_json_file)
        self.vocab_json = load_json(vocab_json_path)
        self.new_vocab_json = load_json(new_vocab_json_path)
        self.DEFAULT_OOV_TOKEN = len(self.vocab_json['label2idx'].keys())
        self.img_dir = img_dir
        self.scenegraphs_json = scenegraphs_json
        self.sgg_features_h5_file = sgg_features_h5_file
        
        self.GQA_with_scene_graph = GQASceneDataset(self.scenegraphs_json, self.vocab_json, self.embedding_json)
        self.new_vocab_len = len(self.new_vocab_json['word2idx'].keys())

        self.tgt_seq_len = tgt_seq_len
        self.num_bins = num_bins

        # count how many images are loaded
        self.count = 0

    def __getitem__(self, idx):
        self.count += 1

        # generating imageid
        keys = list(self.scenegraphs_json.keys())
            
        imageid = keys[idx]
        
        required_len = self.tgt_seq_len
        tgt_seq, scenegraph = self.GQA_with_scene_graph.SceneGraph2SeqV2(imageid, self.num_bins, required_len)
        
        src_text = 'Parse image.'
        

        return tgt_seq, imageid, src_text


    def __len__(self):
        keys = list(self.scenegraphs_json.keys())
        return len(keys)

