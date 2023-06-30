from inflection import singularize as inf_singularize
from pattern.text.en import singularize as pat_singularize




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


class GQASceneDataset(Dataset) :
    def __init__(self, scenegraphs_json, vocab_json, embedding_json) :
        # print("scenegraphs_json_path: ", scenegraphs_json_path)
        self.scenegraphs_json = scenegraphs_json
        self.vocab_json = vocab_json
        self.embedding_json = embedding_json

        self.len_labels = len(self.vocab_json['label2idx'])
        self.len_attr = len(self.vocab_json['attr2idx'])

        self.embed_size = 300
        self.attr_length = 622


    def getname2idx(self, name) :
        try :
            cid = self.vocab_json['label2idx'][name]
        except :
            try :
                try:
                    name1 = inf_singularize(name)
                    cid = self.vocab_json['label2idx'][name1]
                except:
                    name2 = pat_singularize(name)
                    cid = self.vocab_json['label2idx'][name2]
            except :
                name = name.rstrip('s')
                cid = self.vocab_json['label2idx'][name]
        return cid

    def getattr2idx(self, attr) :
        try :
            cid = self.vocab_json['attr2idx'][attr]
        except :
            try:
                try:
                    attr1 = inf_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr1]
                except:
                    attr2 = pat_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr2]
            except :
                name = attr.rstrip('s')
                cid = self.vocab_json['label2idx'][name]

        return cid

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


    def scene2embedding(self, imageid) :
        #print (imageid)
        meta = dict()
        # embeds = dict()
        
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        objects = []
        objects_name = []
        objects_attr = []
        boxes = [[0,0,0,0]]*36
        labels_embeddings = []
        attr_embeddings = []
        boxes_embed = []

        for i,obj_id in enumerate(info_raw['objects'].keys()) :
            # embeds[obj_id] = {}
            obj = info_raw['objects'][obj_id]
            obj_name = np.zeros(self.len_labels, dtype=np.float32)
            obj_attr = np.zeros(self.len_attr, dtype=np.float32)
            box = np.zeros(4, dtype=np.float32)
            name = obj['name']
            # embeds[obj_id]['name'] = name

            try : 
                cid = self.getname2idx(name)
                label_embed = self.getembedding(cid, is_label=True)
                labels_embeddings.append(label_embed)
                obj_name[cid] = 1

                # embeds[obj_id]['attr_embed'] = []
                for attr in obj['attributes'] :
                    if not attr:
                        attr_embed = 0
                    else:
                        cid = self.getattr2idx(attr)
                        attr_embed = self.getembedding(cid)
                    attr_embeddings.append(attr_embed)
                    obj_attr[cid] = 1
                #pdb.set_trace()
                #objects_name.append(obj_name)
                #objects_attr.append(obj_attr)

                box[0] = abs(float(obj['x']))
                box[1] = abs(float(obj['y']))
                box[2] = abs(float(obj['x'] + obj['w']))
                box[3] = abs(float(obj['y'] + obj['h']))
                boxes[i] = box

            except :
                continue

        zero_embedding = np.asarray([0]*300)
        if len(labels_embeddings) < 36:
            for i in range(36 - len(labels_embeddings)):
                labels_embeddings.append(zero_embedding)
        else:
            labels_embeddings = labels_embeddings[:36] 
        
        if len(attr_embeddings) < 36:
            for i in range(36 - len(attr_embeddings)):
                attr_embeddings.append(zero_embedding)
        else:
            attr_embeddings = attr_embeddings[:36] 
        #embeddings = labels_embeddings + attr_embeddings
        #len_embedding = len(embeddings)
        out = np.zeros((36,300))
        for i in range(36) :
           out[i] = np.add(labels_embeddings[i], attr_embeddings[i])

        return out, boxes
        
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

    def SceneGraph2SeqV2(self, imageid, num_bins, required_len = None):
        '''
        version 2.0 GQA dataset
        Sequence example:
        <obj0_name> is <R01> <obj1_name> . 
        '''
        scenegraph = self.scenegraphs_json[imageid]
        seq = []
        objid2bbox = dict()
        objid2name = dict()

        # get the map between obj_id and their names
        for obj_id in scenegraph['objects'].keys():
            obj0 = scenegraph['objects'][obj_id]
            name0 = obj0['name']
            objid2name[obj_id] = name0
            x01 = "<bin_{}>".format(int((obj0['x'] * (num_bins - 1)).round()))
            y01 = "<bin_{}>".format(int((obj0['y'] * (num_bins - 1)).round()))
            x02 = "<bin_{}>".format(int(((obj0['x'] + obj0['w']) * (num_bins - 1)).round()))
            y02 = "<bin_{}>".format(int(((obj0['y'] + obj0['h']) * (num_bins - 1)).round()))
            bbox0 = []
            bbox0.append(x01)
            bbox0.append(y01)
            bbox0.append(x02)
            bbox0.append(y02)
            objid2bbox[obj_id] = bbox0

        for i, obj_id in enumerate(scenegraph['objects'].keys()):
            obj = scenegraph['objects'][obj_id]
            
            if len(obj['relations']) != 0:
                name = obj['name']
                seq.append(name)
                seq.append(str(obj['x']))
                seq.append(str(obj['y']))
                seq.append(str(obj['w']))
                seq.append(str(obj['h']))
                seq.append('is')
                
                for i, rel in enumerate(obj['relations']):
                    rel_name = rel['name']
                    rel_names = rel_name.split(' ')
                    for j in range(len(rel_names)):
                        seq.append(rel_names[j])
                    obj2_id = rel['object']
                    obj2_name = objid2name[obj2_id]
                    seq.append(obj2_name)
                    obj2_bbox = objid2bbox[obj2_id]
                    seq.append(obj2_bbox[0])
                    seq.append(obj2_bbox[1])
                    seq.append(obj2_bbox[2])
                    seq.append(obj2_bbox[3])
                    if i < len(obj['relations'])-1:
                        seq.append(',')
                    else:
                        seq.append('.')
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

        return seq, scenegraph

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

def relation()
    flag = [0, 0, 0]
    for i in range(len(relation_raw)):
        if flag == [1, 0, 0]:
            continue
        elif flag == [1, 1, 0]:
            flag = [1, 0, 0]
            continue
        elif flag == [1, 1, 1]:
            flag = [1, 1, 0]
            continue

        if relation_raw[i] in dataset.ind_to_predicates:
            relation.append(dataset.ind_to_predicates[relation_raw[i]])

        elif i != len(relation_raw)-1 and ((relation_raw[i]+' '+relation_raw[i+1]) in dataset.ind_to_predicates):
            r = relation_raw[i]+' '+relation_raw[i+1]
            relation.append(dataset.ind_to_predicates[r])
            flag = [1, 0, 0]

        elif i != len(relation_raw)-2 and ((relation_raw[i]+' '+relation_raw[i+1]+' '+relation_raw[i+2]) in dataset.ind_to_predicates):
            r = relation_raw[i]+' '+relation_raw[i+1]+' '+relation_raw[i+2]
            relation.append(dataset.ind_to_predicates[r])
            flag = [1, 1, 0]

        elif i != len(relation_raw)-3 and ((relation_raw[i]+' '+relation_raw[i+1]+' '+relation_raw[i+2]+' '+relation_raw[i+3]) in dataset.ind_to_predicates):
            r = relation_raw[i]+' '+relation_raw[i+1]+' '+relation_raw[i+2]+' '+relation_raw[i+3]
            relation.append(dataset.ind_to_predicates[r])
            flag = [1, 1, 1]

        else:
            print("Not find relation: ", relation_raw[i])

            # TODO: convert to relation tuple


def target2seq_raw(self, image, target, required_len=None):

        seq = []
        obj_num = target.bbox.shape[0]
        (w, h) = target.size
        for i in range(obj_num): # each object
            bbox = target.bbox[i, :].unsqueeze(0)
            boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
            boxes_target["boxes"] = bbox
            boxes_target["labels"] = np.array([0])
            boxes_target["area"].append((float(bbox[:,2]) - float(bbox[:,0])) * (float(bbox[:,3]) - float(bbox[:,1])))
            boxes_target["area"] = torch.tensor(boxes_target["area"])
            patch_image, bbox_new = self.positioning_transform(image, boxes_target)
            bbox_value = bbox_new['boxes']
            # print("bbox after transform: ", [t.detach().numpy() for t in bbox_new['boxes']])
            # pdb.set_trace()




            # # for debugging
            # bbox = torch.tensor([200/w, 200/h, 300/w, 300/h])
            # patch_image, _ = self.detection_transform(image, boxes_target)

            # print("bbox_value: ", bbox_value)
            # pdb.set_trace()


            # print("bbox: ", bbox)
            obj_id = target.extra_fields['labels'][i]
            attrs_id = target.extra_fields['attributes'][i, :]
            relations_id = target.extra_fields['relation'][i, :]
            rel_num = np.count_nonzero(relations_id)
            if rel_num == 0:
                continue
            else:
                obj_name = self.ind_to_classes[obj_id]
                seq.append(self.bpe.encode(obj_name + '&&'))
                # seq.append(self.bpe.encode(obj_name))
                seq.append("<bin_{}>".format(str(round(bbox_value[0][0].item() * (self.num_bins - 1)))))
                seq.append("<bin_{}>".format(str(round(bbox_value[0][1].item() * (self.num_bins - 1)))))
                seq.append("<bin_{}>".format(str(round(bbox_value[0][2].item() * (self.num_bins - 1)))))
                seq.append("<bin_{}>".format(str(round(bbox_value[0][3].item() * (self.num_bins - 1)))))
                seq.append(self.bpe.encode('&&is&&'))
                # seq.append(self.bpe.encode('is'))
                rel_cnt = 0
                for j in range(obj_num):
                    if relations_id[j] != 0:
                        rel_cnt += 1
                        obj2_id = target.extra_fields['labels'][j]
                        obj2_name = self.ind_to_classes[obj2_id]
                        rel_id = relations_id[j]
                        rel_name = self.ind_to_predicates[rel_id]
                        rel_name_words = rel_name.split(' ')
                        for k in range(len(rel_name_words)):
                            seq.append(self.bpe.encode(rel_name_words[k] + '&&'))
                            # seq.append(self.bpe.encode(rel_name_words[k]))
                        seq.append(self.bpe.encode(obj2_name + '&&'))
                        # seq.append(self.bpe.encode(obj2_name))
                        bbox2 = target.bbox[j, :]
                        seq.append("<bin_{}>".format(str(round(bbox2[0][0].item() * (self.num_bins - 1)))))
                        seq.append("<bin_{}>".format(str(round(bbox2[0][1].item() * (self.num_bins - 1)))))
                        seq.append("<bin_{}>".format(str(round(bbox2[0][2].item() * (self.num_bins - 1)))))
                        seq.append("<bin_{}>".format(str(round(bbox2[0][3].item() * (self.num_bins - 1)))))
                        if rel_cnt < rel_num:
                            seq.append(self.bpe.encode('&&,&&'))
                            # seq.append(self.bpe.encode(','))
                        else:
                            seq.append(self.bpe.encode('&&.&&'))
                            # seq.append(self.bpe.encode('.'))
        
        seq_len = len(seq)
        if required_len:
            if seq_len > required_len:
                seq = seq[:required_len]
        
        return seq


# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# y_true = ["cat", "dog", "dog", "cat", "rabbit"]
# y_pred = ["cat", "cat", "dog", "cat", "rabbit"]

# # Calculate accuracy
# accuracy = accuracy_score(y_true, y_pred)
# print("Accuracy:", accuracy)

# # Calculate precision
# precision = precision_score(y_true, y_pred, average="macro")
# print("Precision:", precision)

# # Calculate recall
# recall = recall_score(y_true, y_pred, average="macro")
# print("Recall:", recall)

# # Calculate F1 score
# f1 = f1_score(y_true, y_pred, average="macro")
# print("F1 score:", f1)