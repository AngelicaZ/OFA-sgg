import os
from re import X
from unicodedata import name
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
from torch.utils.data import Dataset
import utils.transforms as T


def load_json(fname) :
    json_dict = json.load(open(fname))

    return json_dict

def np_load(fname) :
    np_array = np.load(fname)

    return np_array

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class GQASceneDataset(Dataset) :
    def __init__(self, 
                 scenegraphs_json, 
                 vocab_json, 
                 embedding_json,
                 patch_image_size=512,
                 max_image_size=512,
                 imagenet_default_mean_and_std=False) :
        # print("scenegraphs_json_path: ", scenegraphs_json_path)
        self.scenegraphs_json = scenegraphs_json
        self.vocab_json = vocab_json
        self.embedding_json = embedding_json

        self.len_labels = len(self.vocab_json['label2idx'])
        self.len_attr = len(self.vocab_json['attr2idx'])

        self.embed_size = 300
        self.attr_length = 622

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])

    def box2embedding(self, box) :
        proj = nn.Linear(4, self.embed_size)
        box = torch.from_numpy(box)
        embed = proj(box)
        return embed

    def getembedding(self, cid, is_label=False) :
        embed = np.empty(self.embed_size)
        if is_label :
            embed = self.embedding_json[self.attr_length + cid]
        else :
            embed = self.embedding_json[cid]
        embed = [float(emb) for emb in embed]
        embed = np.asarray(embed)
        return embed


    
        
    def extractembeddings(self,images_list, mapping) :
        final_embeddings = mapping
        images = json.load(open(images_list))
        i=0
        for image in images :
            embeddings, bboxes = self.scene2embedding(image)
            #embeddings = embeddings.astype(np.double)
            final_embeddings[image] = {}
            final_embeddings[image]['objandattr'] = embeddings
            final_embeddings[image]['bboxes'] = bboxes
           
           # i += 1
           # if i>250 :
           #     break
        return final_embeddings

    def SceneGraph2SeqV2(self, imageid, image, num_bins, required_len = None, obj_order = False):
        '''
        version 2.0 GQA dataset
        Sequence example:
        <obj0_name> is <R01> <obj1_name> . 
        '''
        scenegraph = self.scenegraphs_json[imageid]
        seq = []
        objid2bbox = dict()
        objid2name = dict()
        objid2area = dict()

        bbox_seq = []
        obj_num = len(list(scenegraph['objects'].keys()))
        w = scenegraph['width']
        h = scenegraph['height']

        # normalize the bboxes
        bbox = torch.zeros((len(list(scenegraph['objects'].keys())), 4))
        for i, obj_id in enumerate(list(scenegraph['objects'].keys())):
            obj = scenegraph['objects'][obj_id]
            x1 = obj['x']
            y1 = obj['y']
            x2 = obj['x'] + obj['w']
            y2 = obj['y'] + obj['h']
            bbox[i, 0] = x1
            bbox[i, 1] = y1
            bbox[i, 2] = x2
            bbox[i, 3] = y2

        region_raw = []
        for i in range(obj_num):
            region_raw.extend(bbox[i, :])
        region = torch.tensor(region_raw)

        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        boxes_target["boxes"] = bbox
        boxes_target["labels"] = np.array([0])
        for i in range(obj_num):
            boxes_target["area"].append((float(bbox[i,2]) - float(bbox[i,0])) * (float(bbox[i,3]) - float(bbox[i,1])))

        boxes_target["area"] = torch.tensor(boxes_target["area"])
        patch_image, bbox_new = self.positioning_transform(image, boxes_target)
        bbox_value = bbox_new['boxes']
        resize_h, resize_w = bbox_new["size"][0], bbox_new["size"][1]
        w_resize_ratio = resize_w / w
        h_resize_ratio = resize_h / h


        # get the map between obj_id and their names
        for i, obj_id in enumerate(list(scenegraph['objects'].keys())):
            obj0 = scenegraph['objects'][obj_id]
            name0 = obj0['name']
            objid2name[obj_id] = name0
            x01 = "<bin_{}>".format(int((bbox_value[i][0].item() * (num_bins - 1)).round()))
            y01 = "<bin_{}>".format(int((bbox_value[i][1].item() * (num_bins - 1)).round()))
            x02 = "<bin_{}>".format(int((bbox_value[i][2].item() * (num_bins - 1)).round()))
            y02 = "<bin_{}>".format(int((bbox_value[i][3].item() * (num_bins - 1)).round()))
            bbox0 = []
            bbox0.append(x01)
            bbox0.append(y01)
            bbox0.append(x02)
            bbox0.append(y02)
            objid2bbox[obj_id] = bbox0
            objid2area[obj_id] = obj0['w'] * obj0['h']
        
        # rearrange the objects from big to small
        if obj_order:
            new_area_list = list(objid2area.values())
            new_area_list.sort(reverse=True)
            objid_sort = []
            for i in range(obj_num):
                new_objid = new_area_list.index(list(objid2area.values())[i])
                objid_sort.append(new_objid)


        for m in range(obj_num):
            if obj_order:
                i = objid_sort[m]
            else:
                i = m
            
            # For PredCls,add obj labels
            obj_id = list(scenegraph['objects'].keys())[i]
            obj_name = obj_id['name']
            bbox_seq_i.append(obj_name)

            bbox_seq_i = []
            x1 = round(bbox_value[i][0].item() * (self.num_bins - 1))
            y1 = round(bbox_value[i][1].item() * (self.num_bins - 1))
            x2 = round(bbox_value[i][2].item() * (self.num_bins - 1))
            y2 = round(bbox_value[i][3].item() * (self.num_bins - 1))

            bbox_vals_normalize = [x1, y1, x2, y2]
            for n,val in enumerate(bbox_vals_normalize):
                if val < 0:
                    bbox_vals_normalize[n] = abs(bbox_vals_normalize[n])
                if val > self.num_bins - 1:
                    bbox_vals_normalize[n] = self.num_bins - 1


            # For PredCls and SGCls, add bbox
            bbox_seq_i.append("<bin_{}>".format(str(bbox_vals_normalize[0])))
            bbox_seq_i.append("<bin_{}>".format(str(bbox_vals_normalize[1])))
            bbox_seq_i.append("<bin_{}>".format(str(bbox_vals_normalize[2])))
            bbox_seq_i.append("<bin_{}>".format(str(bbox_vals_normalize[3])))
            
            if m < obj_num - 1:
                bbox_seq_i.append(',')
            else:
                bbox_seq_i.append('.')
            bbox_seq_i = ' '.join(bbox_seq_i)
            # print("bbox_seq_i: ", bbox_seq_i)
            bbox_seq.append(bbox_seq_i)

        

        for m, obj_id in enumerate(scenegraph['objects'].keys()):
            if obj_order:
                i = objid_sort[m]
            else:
                i = m

            obj = scenegraph['objects'][obj_id]
            
            if len(obj['relations']) != 0:
                name = obj['name']
                seq.append(self.bpe.encode(name + '&&'))
                seq.append(str(obj['x']))
                seq.append(str(obj['y']))
                seq.append(str(obj['w']))
                seq.append(str(obj['h']))
                seq.append(self.bpe.encode('&&is&&'))
                
                for j, rel in enumerate(obj['relations']):
                    rel_name = rel['name']
                    rel_names = rel_name.split(' ')
                    for k in range(len(rel_names)):
                        seq.append(self.bpe.encode(rel_names[k] + '&&'))
                    obj2_id = rel['object']
                    obj2_name = objid2name[obj2_id]
                    seq.append(obj2_name)
                    obj2_bbox = objid2bbox[obj2_id]
                    seq.append(obj2_bbox[0])
                    seq.append(obj2_bbox[1])
                    seq.append(obj2_bbox[2])
                    seq.append(obj2_bbox[3])
                    if j < len(obj['relations'])-1:
                        seq.append(self.bpe.encode('&&,&&'))
                    else:
                        seq.append(self.bpe.encode('&&.&&'))
            # else:
            #     seq.append('has')
            #     seq.append('no')
            #     seq.append('relation')
            #     seq.append('.')

        seq_len = len(seq)
        # count_len_path = '/home/chenyu/scene_graph_generation/OFA/count_tgt_seq_len.txt'
        # with open(count_len_path, 'a') as f:
        #     f.write(f"length: {seq_len}\n") # len = 125, 133
        if required_len:
            if seq_len > required_len:
                seq = seq[:required_len]

        return seq, patch_image, bbox_seq, w_resize_ratio, h_resize_ratio, region, scenegraph

    def SceneGraph2Seq(self, imageid, required_len = None):
        """
        version 1.0
        Sequence example:
        <START> <width> <height> <'obj0'> <obj0id> <x> <y> <w> <h> <attr0> <attr1> ... <'obj1'> <obj1id> ... 
        <'obj2'> <obj2id> ... <'R01'> <relation0> <'R10'> <relation1> ... <END> <PAD> <PAD> ...

        Light version:
        <START> <obj0> <obj0_name> ... <obj1> <obj1_name> ... <obj2> <obj2id> ... <R01> <relation0> <R10> <relation1> ... <END> <PAD> <PAD> ...
        """
        scenegraph = self.scenegraphs_json[imageid]
        seq = []
        relation = dict()
        obj2idx = dict()
        seq.append('<START>')
        for i, obj_id in enumerate(scenegraph['objects'].keys()) :
            seq.append(f'obj_{i}')
            
            obj2idx[obj_id] = i
            obj = scenegraph['objects'][obj_id]
            name = obj['name']
            seq.append(name)
            # x = obj['x']
            # y = obj['y']
            # w = obj['w']
            # h = obj['h']
            # seq.append(x)
            # seq.append(y)
            # seq.append(w)
            # seq.append(h)
            # for attr in obj['attributes'] :
            #     seq.append(attr)
        
        for obj_id in scenegraph['objects'].keys():
            obj1 = scenegraph['objects'][obj_id]
            # print("obj1 id: ", obj1)
            idx1 = obj2idx[obj_id]
            for rel in obj1['relations']:
                rel_name = rel['name']
                obj2_id = rel['object']
                idx2 = obj2idx[obj2_id]
                rel_label = f'R_{idx1}_{idx2}'
                relation[rel_label] = rel_name
        
        for key, val in relation.items():
            seq.append(key)
            seq.append(val)
                
        seq.append('<END>')    

        # padding
        # TODO: if less than required lenth, cut off
        # TODO: dump a file to analysis the sequence length
        seq_len = len(seq)
        # print("seq len: ", seq_len)
        if required_len:
            # assert required_len > seq_len
            if required_len > seq_len:
                for i in range(required_len-seq_len):
                    seq.append('<PAD>')
            else:
                seq = seq[:required_len]

        return seq, scenegraph
    

    def generate_new_vocab(self, new_json_path):
        # max obj number
        max_obj_num = 100

        # set up offset
        offset = 0

        # set up a new dict to store
        # Uniform numbers
        new_vocab = dict()
        new_vocab['idx2label'] = self.vocab_json['idx2label']
        new_vocab['label2idx'] = self.vocab_json['label2idx']

        label_len = len(new_vocab['idx2label'].keys())
        offset += label_len
        new_vocab['attr2idx'] = self.vocab_json['attr2idx']
        new_vocab['idx2attr'] = dict()
        for key, val in new_vocab['attr2idx'].items():
            new_vocab['attr2idx'][key] = val + offset
        for key, val in new_vocab['attr2idx'].items():
            new_vocab['idx2attr'][val] = key 

        attr_len = len(new_vocab['idx2attr'].keys())
        offset += attr_len
        new_vocab['rel2idx'] = self.vocab_json['rel2idx']
        new_vocab['idx2rel'] = dict()
        for key, val in new_vocab['rel2idx'].items():
            new_vocab['rel2idx'][key] = val + offset
        for key, val in new_vocab['rel2idx'].items():
            new_vocab['idx2rel'][val] = key
        
        rel_len = len(new_vocab['idx2rel'].keys())
        offset += rel_len
        new_vocab['rellabel2idx'] = dict()
        new_vocab['idx2rellabel'] = dict()
        count = 0
        for i in range(max_obj_num): # maximum 100 objects
            for j in range(max_obj_num):
                new_vocab['idx2rellabel'][count+offset] = f'R{i}_{j}'
                count += 1
        for key, val in new_vocab['idx2rellabel'].items():
            new_vocab['rellabel2idx'][val] = key
        
        rellabel_len = len(new_vocab['idx2rellabel'].keys())
        offset += rellabel_len
        new_vocab['objlabel2idx'] = dict()
        new_vocab['idx2objlabel'] = dict()
        for i in range(max_obj_num):
            new_vocab['idx2objlabel'][i+offset] = f'obj{i}'
        for key, val in new_vocab['idx2objlabel'].items():
            new_vocab['objlabel2idx'][val] = key


        objlabel_len = len(new_vocab['idx2objlabel'].keys())
        offset += objlabel_len
        new_vocab['specialannotation2idx'] = dict()
        new_vocab['idx2specialannotation'] = dict()
        new_vocab['idx2specialannotation'][0+offset] = '<START>'
        new_vocab['idx2specialannotation'][1+offset] = '<END>'
        new_vocab['idx2specialannotation'][2+offset] = '<PAD>'
        new_vocab['specialannotation2idx']['<START>'] = 0+offset
        new_vocab['specialannotation2idx']['<END>'] = 1+offset
        new_vocab['specialannotation2idx']['<PAD>'] = 2+offset

        json_object = json.dumps(new_vocab, indent=4)

        with open(new_json_path, 'w') as f:
            f.write(json_object)

    def generate_new_vocab_without_categories(self, new_json_path):
        # max obj number
        max_obj_num = 100

        # set up offset
        offset = 0

        # set up a new dict to store
        # Uniform numbers
        new_vocab = dict()
        new_vocab['idx2word'] = dict()
        new_vocab['word2idx'] = dict()
        
        for key, val in self.vocab_json['label2idx'].items():
            new_vocab['word2idx'][key] = val + offset

        offset = len(new_vocab['word2idx'].keys())
        for key, val in self.vocab_json['attr2idx'].items():
            new_vocab['word2idx'][key] = val + offset
         

        offset = len(new_vocab['word2idx'].keys())
        for key, val in self.vocab_json['rel2idx'].items():
            new_vocab['word2idx'][key] = val + offset
        
        offset = len(new_vocab['word2idx'].keys())
        idx2rellabel = dict()
        count = 0
        for i in range(max_obj_num): # maximum 100 objects
            for j in range(max_obj_num):
                idx2rellabel[count+offset] = f'R{i}_{j}'
                count += 1
        for key, val in idx2rellabel.items():
            new_vocab['word2idx'][val] = key
        
        offset = len(new_vocab['word2idx'].keys())
        idx2objlabel = dict()
        for i in range(max_obj_num):
            idx2objlabel[i+offset] = f'obj{i}'
        for key, val in idx2objlabel.items():
            new_vocab['word2idx'][val] = key

        # x, y, w, h
        offset = len(new_vocab['word2idx'].keys())
        xywh = dict()
        for i in range(1024):
            xywh[i+offset] = i
        for key, val in xywh.items():
            new_vocab['word2idx'][val] = key


        offset = len(new_vocab['word2idx'].keys())
        new_vocab['word2idx']['<START>'] = 0+offset
        new_vocab['word2idx']['<END>'] = 1+offset
        new_vocab['word2idx']['<PAD>'] = 2+offset

        for key, val in new_vocab['word2idx'].items():
            new_vocab['idx2word'][val] = key

        json_object = json.dumps(new_vocab, indent=4)

        with open(new_json_path, 'w') as f:
            f.write(json_object)  
