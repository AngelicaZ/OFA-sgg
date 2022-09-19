
import json
import pdb
from posixpath import split
import time
from turtle import pd, shape
from matplotlib import image
import numpy as np
from data.mm_data.sg_raw import GQASceneDataset, load_json, np_load
import h5py
import logging
from typing import Iterable, Iterator, Callable, Dict
import torch
from torch.utils.data import Dataset
from inflection import singularize as inf_singularize
from pattern.text.en import singularize as pat_singularize
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
        
        # scene_graph = self.sg_features_json[imageid]
        # index = scene_graph['index']
        # width = scene_graph['width']
        # height = scene_graph['height']
        # objectsNum = scene_graph['objectsNum']
        # with h5py.File(self.sgg_features_h5_file, "r") as f:
        #     # List all groups
        #     all_scores_key = list(f.keys())[0]
        #     attributes_key = list(f.keys())[1]
        #     bboxes_key = list(f.keys())[2]
        #     features_key = list(f.keys())[3]
        #     labels_key = list(f.keys())[4]
        #     scores_key = list(f.keys())[5]

        #     # Get the data, use the feature and bbox to calculate the final features
        #     features = f[features_key][index][:36]  # only take the first 36 objects,  (36, 2048)
        #     bboxes = f[bboxes_key][index][:36] # size(36, 4), (x1, y1, x2, y2)
        #     obj_ids = f[labels_key][index][:36]
        #     src_features = features # [36, 2048]
        
        required_len = self.tgt_seq_len
        tgt_seq, scenegraph = self.GQA_with_scene_graph.SceneGraph2SeqV2(imageid, self.num_bins, required_len)
        # print("tgt seq: ", tgt_seq)
        # print("scene graph: ", scenegraph)
        
        # pdb.set_trace()
        # tgt_seq_len = len(tgt_seq)
        # tgt_seq_ids = np.zeros([tgt_seq_len])
        # for i, word in enumerate(tgt_seq):
        #     tgt_seq_ids[i] = self.one_hot_encode(word)
        
        src_text = 'Parse image.'
        

        return tgt_seq, imageid, src_text


    def __len__(self):
        keys = list(self.scenegraphs_json.keys())
        return len(keys)
    
    def one_hot_encode(self, name):
        name = str(name)
        vocab = self.new_vocab_json
        # embed = np.zeros([1, self.new_vocab_len])
        # print("word: ", name)
        try:
            try:
                try:
                    idx = vocab['word2idx'][name]
                except :
                    try :
                        try:
                            name1 = inf_singularize(name)
                            idx = vocab['word2idx'][name1]
                        except:
                            name2 = pat_singularize(name)
                            idx = vocab['word2idx'][name2]
                    except :
                        name = name.rstrip('s')
                        idx = vocab['word2idx'][name]
                
            except:
                name3 = name + "ed"
                # print("name plus ed: ", name)
                idx = vocab['word2idx'][name3]
            embed = idx
        except:
            print("Word no found: ", name)
            embed = self.DEFAULT_OOV_TOKEN

        return embed
    
    def features2seq(self, obj_ids, bboxes, features, required_len = None):
        """
        Sequence example:
        <width> <height> <'obj0'> <obj0id> <x> <y> <w> <h> <feature> <attr0> <attr1> ... <'obj1'> <obj1id> ... 
        <'obj2'> <obj2id> ... <'R01'> <relation0> <'R10'> <relation1> ...

        Light version:
        <obj0> <obj0id> <x> <y> <w> <h> <feature> <obj1> <obj1id> ... <obj2> <obj2id> ... 
        """
        seq = []
        for i, obj_id in enumerate(obj_ids):
            seq.append(f'obj{i}')
            seq.append(obj_id)
            bbox = bboxes[i] # (x1, y1, x2, y2)
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            seq.append(x)
            seq.append(y)
            seq.append(w)
            seq.append(h)
            feature = features[i]
            seq.append(feature)
        
        # padding
        if required_len:
            seq_len = len(seq)
            assert required_len > seq_len
            for i in range(required_len-seq_len):
                seq.append('<PAD>')
        
        return seq



    def print_new_vocab(self, new_vocab_path):
        self.GQA_with_scene_graph.generate_new_vocab_without_categories(new_vocab_path)

    def get_obj_idx(self, obj):
        try:
            return self.GQA_with_scene_graph.getname2idx(obj)
        except:
            return self.DEFAULT_OOV_TOKEN


    def Calculate_source_att_map(self, bboxes, bboxesSG):
        # bbox: (x1, y1, x2, y2)
        # bboxSG: (x1, y1, x2, y2)
        threshold = 0.85
        map = []
        for i, bbox in enumerate(bboxes):
            iou_list = []
            for bboxSG in bboxesSG:
                iou = self.calculate_iou(bbox, bboxSG)
                iou_list.append(iou)
            iou_max = max(iou_list)
            if iou_max >= threshold:
                iou_max_idx = iou_list.index(iou_max)
                map_i = (i, iou_max_idx)
                map.append(map_i)
            else:
                map.append((np.NaN, np.NaN)) # TODO: is it correct?
        
        return map
    
    def calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def Calculate_target_att_map(self, bboxesSG):
        # print("bbox_sg: ", bboxesSG)
        map = []
        # print("size: ", bboxesSG.size())
        for i in range(len(bboxesSG)):
            map.append((i, i))
        
        return map

    def prepare_scene_graph_properties(self, imageid, scene_graph_ground_truth, vocab_json) -> dict:
    
        """
        An example of scene graph

        {
            "2407890": {
                "width": 640,
                "height": 480,
                "location": "living room",
                "weather": none,
                "objects": {
                    "271881": {
                        "name": "chair",
                        "x": 220,
                        "y": 310,
                        "w": 50,
                        "h": 80,
                        "attributes": ["brown", "wooden", "small"],
                        "relations": {
                            "32452": {
                                "name": "on",
                                "object": "275312"
                            },
                            "32452": {
                                "name": "near",
                                "object": "279472"
                            }                    
                        }
                    }
                }
            }
        }
        """

        # data = {}
        # start_time = time.time()
        # print("Start to load scene graphs from %s" % scenegraphs_json_path)

        
        
        tgt_attr_vocab = vocab_json['attr2idx']
        edge_attr_vocab = vocab_json['rel2idx']
        

        
        item = dict()
        embeds, boxes = self.GQA_with_scene_graph.scene2embedding(imageid) 
        item["img_id"] = imageid

        info_raw = scene_graph_ground_truth
        item["img_h"] = int(info_raw['height'])
        item["img_w"] = int(info_raw['width'])

        objects_id = []
        attrs_id = []
        edge_heads = []
        edge_types = []
        tgt_objs = []

        tgt_attr = []
        edge_attr = []
        
        
        is_related = dict() 
        for i, obj_id in enumerate(info_raw['objects']):

            # prepare target attr and edge attr
            tgt_attr_per_obj = dict()
            edge_attr_per_obj = dict()
            for attr_name in tgt_attr_vocab.keys():
                tgt_attr_per_obj[attr_name] = 0
            for rel_name in edge_attr_vocab.keys():
                edge_attr_per_obj[rel_name] = 0

            
            obj = info_raw['objects'][obj_id]
            name = obj['name']
            tgt_objs.append(name)

            # calculate obj_id/attr_id
            try:
                obj_cid = self. GQA_with_scene_graph.getname2idx(name)
            except:
                obj_cid = 0
            objects_id.append(int(obj_cid))
            for attr in obj['attributes'] :
                tgt_attr_per_obj[attr] = 1
                try:
                    attr_cid = self.GQA_with_scene_graph.getattr2idx(attr)
                except:
                    attr_cid = 0
                attrs_id.append(attr_cid)
            
            # calculate edge heads
            if len(obj["relations"]) != 0:
                relations = obj["relations"]
                # print("relations: ", relations)
                for relation in relations: # relations is a list
                    # print("key: ", key)
                    # relation = relations[key]
                    # sprint("relation: ", relation)
                    related_obj_id = relation["object"]
                    edge_attr_per_obj[relation["name"]] = 1
                    if  related_obj_id not in is_related.keys():
                        is_related[related_obj_id] = []
                    if obj_id not in is_related[related_obj_id]:
                        edge_heads.append(int(i+1))  # the index of object in each image
                        edge_types.append(relation["name"])
                        if related_obj_id not in is_related.keys():
                            is_related[related_obj_id] = []
                        if obj_id not in is_related.keys():
                            is_related[obj_id] = []
                        is_related[related_obj_id].append(obj_id)
                        is_related[obj_id].append(related_obj_id)
                        
                            
            # if len(info_raw['objects']) > 0: # TODO: why there are pics without objects?
            tgt_attr.append(tgt_attr_per_obj)
            edge_attr.append(edge_attr_per_obj)
            

        # objects_id is the one-hot encoder
        # need to make the object_id of different images to be the same size
        # Truncated and filled (Threshold = 36)
        if len(objects_id) < 36:
            for i in range(36 - len(objects_id)):
                objects_id.append(0)
        else:
            objects_id = objects_id[:36]    

        if len(attrs_id) < 36:
            for i in range(36 - len(attrs_id)):
                attrs_id.append(0)
        else:
            attrs_id = attrs_id[:36]

        item["objects_id"] = objects_id
        item["objects_conf"] = [1]*36
        item["attrs_id"] = attrs_id
        item["attrs_conf"] = [1]*36
        item["num_boxes"] = len(boxes)
        item["boxes"] = boxes
        item["features"] = embeds
        item["edge_heads"] = edge_heads
        item["edge_types"] = edge_types
        item["tgt_objs_names"] = tgt_objs
        item["tgt_attr"] = tgt_attr
        item["edge_attr"] = edge_attr

        num_boxes = item['num_boxes']
        decode_config = [
            ('objects_id', np.int64),
            ('objects_conf',  np.float32),
            ('attrs_id',  np.int64),
            ('attrs_conf',  np.float32),
            ('boxes', np.float32),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            ('features', np.float32), # is the embeddings
        ]
        
        for key, datatype in decode_config:
            item[key] = np.array(item[key], dtype = datatype)
            # item[key] = item[key].reshape(shape)
            item[key].setflags(write=False)
        
        return item

