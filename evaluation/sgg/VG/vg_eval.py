from bdb import set_trace
from gc import get_debug
import logging
import os
import pdb
import torch
import numpy as np
import json
from tqdm import tqdm
from functools import reduce
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import random
from PIL import Image

# from .utils import get_dataset_statistics
# from maskrcnn_benchmark.structures.bounding_box import BoxList
# from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
# from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps
from .sgg_eval import SGRecall, SGNoGraphConstraintRecall, SGZeroShotRecall, SGNGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGNGMeanRecall, SGAccumulateRecall

def do_vg_evaluation(
    cfg,
    dataset,
    predictions_raw,
    output_folder,
    logger,
    iou_types,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load("evaluation/sgg/VG/zeroshot_triplet.pytorch", map_location=torch.device("cpu")).long().numpy()

    attribute_on = cfg.MODEL.ATTRIBUTE_ON # false
    num_attributes = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES # 201
    # extract evaluation settings from cfg
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: # True
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # False
            mode = 'predcls'
        else:
            mode = 'sgcls'
    else:
        mode = 'sgdet'

    num_rel_category = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # 51
    multiple_preds = cfg.TEST.RELATION.MULTIPLE_PREDS # False
    iou_thres = cfg.TEST.RELATION.IOU_THRESHOLD # 0.5
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = dict()
    predictions = dict()
    for pred in predictions_raw:
        if type(pred) != dict:
            continue
        for result_id, prediction in pred.items():
            image_idx = int(result_id.split('_')[1])
            img_info = dataset.get_img_info(image_idx)
            image_width = img_info["width"]
            image_height = img_info["height"]
            # # recover original size which is before transform
            # predictions[image_id] = prediction.resize((image_width, image_height))

            # prediction is a sequence
            # print("prediction: ", prediction)
            predictions[image_idx] = prediction

            groundtruths[image_idx] = dataset.get_groundtruth(image_idx, evaluation=True)
            
            # groundtruths.append(gt)

    save_output(output_folder, groundtruths, predictions, dataset)
    
    result_str = '\n' + '=' * 100 + '\n'
    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_idx, gt in groundtruths.items():
            labels = gt.get_field('labels').tolist() # integer
            boxes = gt.bbox.tolist() # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_idx,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name} 
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
                ],
            'annotations': anns,
        }
        print("number of categories: ", len(dataset.ind_to_classes))
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_idx, prediction in predictions.items():
            # box = prediction.convert('xywh').bbox.detach().cpu().numpy() # xywh
            # score = prediction.get_field('pred_scores').detach().cpu().numpy() # (#objs,)
            # label = prediction.get_field('pred_labels').detach().cpu().numpy() # (#objs,)

            # image_idx = int(result_id.split('_')[1])
            # sample_id = result_id.split('_')[0]
            # img_name = sample_id + ".jpg"
            # img_dir = '/data/c/zhuowan/gqa/data/images/'
            # image_path = img_dir + img_name
            # print("image path: ", image_path)

            prepare_pred = prepare_prediction(prediction, dataset, pred_mode='xywh')
            box, label, score = prepare_pred.get_bbox(), prepare_pred.get_label(), prepare_pred.score # box is xywh

            print("prediction sentence: ", prediction)
            print("predicted labels: ", label)
            print("predicted boxes: ", box)
            
            # ground truth for debug
            img = Image.open(dataset.filenames[image_idx]).convert("RGB")
            gt_debug = groundtruths[image_idx]
            seq, obj_labels, rel_labels = dataset.target2seq_raw(img, gt_debug, required_len=dataset.required_len, obj_order=False, add_bbox=True)
            label_gt_debug = gt_debug.get_field('labels').tolist() # integer
            box_gt_debug = gt_debug.bbox.tolist() # xyxy
            for i in range(len(box_gt_debug)):
                box_gt_debug[i][2] = max(box_gt_debug[i][2] - box_gt_debug[i][0], 0) # convert to xywh
                box_gt_debug[i][3] = max(box_gt_debug[i][3] - box_gt_debug[i][1], 0)
            print("sentences_gt: ", seq)
            print("box_gt: ", box_gt_debug)
            print("label_gt: ", label_gt_debug)

            pdb.set_trace()

            # for predcls, we set label and score to groundtruth
            # if mode == 'predcls':
            #     label = prediction.get_field('labels').detach().cpu().numpy()
            #     score = np.ones(label.shape[0])
            #     assert len(label) == len(box)
            image_idx = np.asarray([image_idx]*len(box))
            # score_gt_debug = np.asarray([1]*len(box))
            cocolike_predictions.append(
                np.column_stack((image_idx, box, score, label))
                )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)
        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        # imgIds = fauxcoco.getImgIds()
        # print("imgIds: ", imgIds)
        # pdb.set_trace()
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAp = coco_eval.stats[1]
        
        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += '=' * 100 + '\n'

    if "relations" in iou_types:
        result_dict = {}
        evaluator = {}
        # tradictional Recall@K
        eval_recall = SGRecall(result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(result_dict)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        # test on no graph constraint zero-shot recall
        eval_ng_zeroshot_recall = SGNGZeroShotRecall(result_dict)
        eval_ng_zeroshot_recall.register_container(mode)
        evaluator['eval_ng_zeroshot_recall'] = eval_ng_zeroshot_recall
        
        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for no graph constraint mean Recall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates, print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes
        
        for groundtruth, prediction in zip(groundtruths.values(), predictions.values()):
            # pred_local = prepare_prediction(prediction, dataset)
            # relation_local = pred_local.relation
            # if len(relation_local) != 0:
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, dataset)
        
        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)
        
        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_zeroshot_recall.generate_print_string(mode)
        result_str += eval_ng_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)
        
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            result_str += eval_pair_accuracy.generate_print_string(mode)
        result_str += '=' * 100 + '\n'


    logger.info(result_str)
    
    if "relations" in iou_types:
        if output_folder:
            torch.save(result_dict, os.path.join(output_folder, 'result_dict.pytorch'))
        return float(np.mean(result_dict[mode + '_recall'][100]))
    elif "bbox" in iou_types:
        return float(mAp)
    else:
        return -1

class prepare_prediction():
    def __init__(self, prediction, dataset, pred_mode='xyxy') -> None:
        super().__init__()
        self.prediction = prediction
        self.dataset = dataset
        self.pred_mode = pred_mode
        self.box = []
        self.label = []
        self.score = []
        self.relation = []

        # print("self.dataset.predicate_to_ind length: ", len(self.dataset.predicate_to_ind))
        # pdb.set_trace()

        '''
        prediction seq example: 
        Cat x y x y is on mat x y x y, inside house x y x y.
        giraffe [0, 0, 498, 296] is has head [272, 15, 196, 381] .

        '''
        box_raw = []
        obj_names = []
        relation_raw = []

        pred_tokens_raw = self.prediction

        pred_tokens_without_bbox = []
        for i, token_raw in enumerate(pred_tokens_raw):
            if type(token_raw) == list:
                try:
                    if type(pred_tokens_raw[i-1]) != list:
                        obj_name = pred_tokens_raw[i-1]
                    else:
                        continue
                    bbox = token_raw  # bbox: xyxy
                    
                    if len(bbox) < 4:
                        # print("Non-compliant bbox in prediction: ", bbox)
                        continue
                    if pred_mode == 'xywh':
                        bbox[2] = max(bbox[2] - bbox[0], 0) # convert to xywh
                        bbox[3] = max(bbox[3] - bbox[1], 0)
                    if len(bbox) > 4:
                        bbox = bbox[:4]   

                    # label
                    if obj_name in self.dataset.class_to_ind.keys():
                        l_idx = self.dataset.class_to_ind[obj_name]
                        self.label.append(l_idx)
                        obj_names.append(obj_name)
                        box_raw.append(bbox)
                    else:
                        continue
                        # print("obj_name: ", obj_name)
                        # print("len(obj_name): ", len(obj_name))
                        # pdb.set_trace()
                        # if len(obj_name) % 2 == 0:
                        #     obj_name_new = obj_name[:len(obj_name)/2]
                        #     print("obj_name_new: ", obj_name_new)
                        #     pdb.set_trace()
                        #     if obj_name_new in self.dataset.class_to_ind.keys():
                        #         l_idx = self.dataset.class_to_ind[obj_name_new]
                        #         self.label.append(l_idx)
                        #         obj_names.append(obj_name_new)
                        #         box_raw.append(bbox)
                        # else:
                        #     continue
                except:
                    print("pred_tokens_raw: ", pred_tokens_raw)
                    print("unexpected token: ", token_raw)
                    print("obj name: ", obj_name)
            else:
                pred_tokens_without_bbox.append(token_raw)
        
        pred_sentences_full = ' '.join(pred_tokens_without_bbox)
        pred_sentences = pred_sentences_full.split('.')
        # print("pred_sentences: ", pred_sentences)
        # pdb.set_trace()

        for pred_sentence in pred_sentences:
            
            try: 
                pred_tokens = []
                pred_subsentences_without_coma = pred_sentence.split(',')
                for i, pred_subsentence in enumerate(pred_subsentences_without_coma):
                    pred_sub_tokens = pred_subsentence.split()
                    pred_tokens.extend(pred_sub_tokens)
                    if i != len(pred_subsentences_without_coma)-1:
                        pred_tokens.append(',')
            except:
                pred_tokens = pred_sentence.split()


            for j, word in enumerate(pred_tokens):

                while '' in pred_tokens:
                    pred_tokens.remove('')

                if len(pred_tokens) == 1:
                    continue
                    
                if '<' in word or '>' in word:
                    continue
                    
                if len(word) == 1:
                    continue

                # bbox or obj label
                if (type(word) == list) or (word in obj_names):
                    continue
                
                # conjunctions
                elif word in ['is','isis',',' ,'.']:
                    continue
                    
                # relations
                else:
                    r_raw_list = []
                    if  'is' in pred_tokens[j-1] or  'isis' in pred_tokens[j-1] or ',' in pred_tokens[j-1]:
                        if j < (len(pred_tokens)-2): 
                            try:
                                # if j == len(pred_tokens) - 1:
                                #     continue
                                while ',' not in pred_tokens[j+1]:
                                    r_raw_list.append(pred_tokens[j])
                                    j += 1
                                    if j >= len(pred_tokens) -1:
                                        break
                                # if len(r_raw_list) == 0:
                                #     continue
                                
                            except:
                                print("pred_tokens: ", pred_tokens)
                                print("pred_tokens[j]: ", pred_tokens[j])
                                print("j: ", j)
                                print("r_raw_list: ", r_raw_list)
                                pdb.set_trace()
                        else:
                            while j<len(pred_tokens)-1: #  and j<len(pred_tokens)-2
                                # print("j: ", j)
                                r_raw_list.append(pred_tokens[j])
                                j += 1
                        r_raw = ' '.join(r_raw_list)
                        
                        if pred_tokens[0] in obj_names:
                            obj0_name = pred_tokens[0]
                            obj1_name = pred_tokens[j]
                            relation_raw.append((obj0_name, obj1_name, r_raw))
                            # print("obj0: ", obj0_name)
                            # print("obj1: ", obj1_name)
                        else:
                            '''
                            bad pred sentence example:
                            Cannot find the objects to relation! Obj name:  sneaker
                            obj_names:  ['building', 'window', 'building', 'street', 'car', 'bus', 'windshield']
                            pred sentence:  building <bin_0><bin_0><bin_997><bin_996> is has window <bin_0><bin_3><bin_997><bin_99
                            6>is . building <bin_2><bin_0><bin_995><bin_996> , on street <bin_0><bin_835><bin_996><bin_996> .. car
                                <bin_0><bin_765><bin_997><bin_996>sitting in front of bus <bin_3><bin_0><bin_995><bin_990> is<bin_0>h
                            as windshield <bin_4><bin_0><bin_995><bin_987> .sneaker <bin_0><bin_855><bin_0><bin_995> is. man <bin_
                            12><bin_0><bin_995><bin_986> istire <bin_0><bin_675><bin_997><bin_996><bin_997> .<bin_0>
                            '''
                            # print("Cannot find the objects to relation! Obj name: ", pred_tokens[0])
                            # print("obj_names: ", obj_names)
                            # print("pred sentence: ", self.prediction)
                            continue
                    
                    else:
                        continue
        
        # print("label: ", self.label)
        # print("relation raw: ", relation_raw)
        
        # bbox
        # print("box_raw: ", box_raw)
        num_box = len(box_raw)
        try:
            assert len(self.label) == num_box
            assert len(self.label) == len(obj_names)
        except:
            print("prediction: ", prediction)
            print("label: ", self.label)
            print("obj_names: ", obj_names)
            print("bbox: ", box_raw)
            pdb.set_trace()
        self.box = np.zeros((num_box, 4))
        for i in range(num_box):
            self.box[i, :] = box_raw[i]
        # print("self.box: ", self.box)
 
        # label
        self.label = np.asarray(self.label)
        
        # relation
        for (obj0_name, obj1_name, r_raw) in relation_raw:
            try:
                index0 = obj_names.index(obj0_name)
                index1 = obj_names.index(obj1_name)
                r_index = self.dataset.predicate_to_ind[r_raw]
                self.relation.append([index0, index1, r_index])
            except:
                try:
                    if len(obj1_name) % 2 == 0:
                        obj1_name = obj1_name[:len(obj1_name)/2]
                        index0 = obj_names.index(obj0_name)
                        index1 = obj_names.index(obj1_name)
                        r_index = self.dataset.predicate_to_ind[r_raw]
                        self.relation.append([index0, index1, r_index])
                except:
                    continue
                    # print("obj0_name: ", obj0_name)
                    # print("obj1_name: ", obj1_name)
                    # print("relation: ", r_raw)
                # pdb.set_trace()
        self.relation = np.asarray(self.relation)

        # scores are all 1
        self.score = np.ones(self.box.shape[0])

        
    
    def get_bbox(self):
        return self.box
    
    def get_label(self):
        return self.label

    

def save_output(output_folder, groundtruths, predictions, dataset):
    if output_folder:
        torch.save({'groundtruths':groundtruths, 'predictions':predictions}, os.path.join(output_folder, "eval_results.pytorch"))

        #with open(os.path.join(output_folder, "result.txt"), "w") as f:
        #    f.write(result_str)
        # visualization information
        visual_info = []
        for image_id, (groundtruth, prediction) in enumerate(zip(groundtruths.values(), predictions.values())):
            img_file = os.path.abspath(dataset.filenames[image_id])
            groundtruth = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(groundtruth.bbox.tolist(), groundtruth.get_field('labels').tolist())
                ]
            
            prediction = [
                [b[0], b[1], b[2], b[3], dataset.categories[l]] # xyxy, str
                for b, l in zip(prepare_prediction(prediction, dataset).get_bbox(),prepare_prediction(prediction, dataset).get_label())
                ]

            visual_info.append({
                'img_file': img_file,
                'groundtruth': groundtruth,
                'prediction': prediction
                })
        with open(os.path.join(output_folder, "0210_visual_info.json"), "w") as f:
            json.dump(visual_info, f)



def evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator, dataset):
    """
    Returns:
        pred_to_gt: Matching from predicate to GT
        pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
        pred_triplet_scores: [cls_0score, relscore, cls1_score]
    """
    #unpack all inputs
    mode = global_container['mode']

    local_container = {}
    local_container['gt_rels'] = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()

    # print("gt_rels: ", local_container['gt_rels'])
    

    # if there is no gt relations for current image, then skip it
    if len(local_container['gt_rels']) == 0:
        return

    local_container['gt_boxes'] = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()                   # (#gt_objs, 4)
    local_container['gt_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()           # (#gt_objs, )

    # ground truth for debug
    # relation = groundtruth.get_field('relation_tuple').long().detach().cpu().numpy()
    # local_container['pred_rel_inds'] = relation
    # local_container['rel_scores'] = np.ones((len(relation), 1))
    # box = groundtruth.convert('xyxy').bbox.detach().cpu().numpy()
    # local_container['pred_boxes'] = box
    # local_container['pred_classes'] = groundtruth.get_field('labels').long().detach().cpu().numpy()
    # local_container['obj_scores'] = np.ones((len(box), 1))


    prepare_pred = prepare_prediction(prediction, dataset)
    box, label, score, relation = prepare_pred.get_bbox(), prepare_pred.get_label(), prepare_pred.score, prepare_pred.relation

    # about relations
    # local_container['pred_rel_inds'] = prediction.get_field('rel_pair_idxs').long().detach().cpu().numpy()  # (#pred_rels, 2)
    # local_container['rel_scores'] = prediction.get_field('pred_rel_scores').detach().cpu().numpy()          # (#pred_rels, num_pred_class)

    # if there is no predict relations for current image, then skip it
    if relation.shape[0] == 0:
        return

    num_pred_class = 51
    try:
        local_container['pred_rel_inds'] = relation[:, :2]
        local_container['rel_scores'] = np.ones((relation.shape[0], num_pred_class))
    except:
        print("relation shape: ", relation.shape)
        pdb.set_trace()

    # about objects
    # local_container['pred_boxes'] = prediction.convert('xyxy').bbox.detach().cpu().numpy()                  # (#pred_objs, 4)
    # local_container['pred_classes'] = prediction.get_field('pred_labels').long().detach().cpu().numpy()     # (#pred_objs, )
    # local_container['obj_scores'] = prediction.get_field('pred_scores').detach().cpu().numpy()              # (#pred_objs, )
    local_container['pred_boxes'] = box
    local_container['pred_classes'] = label
    local_container['obj_scores'] = score

    # to calculate accuracy, only consider those gt pairs
    # This metric is used by "Graphical Contrastive Losses for Scene Graph Parsing" 
    # for sgcls and predcls
    if mode != 'sgdet':
        evaluator['eval_pair_accuracy'].prepare_gtpair(local_container)

    # to calculate the prior label based on statistics
    evaluator['eval_zeroshot_recall'].prepare_zeroshot(global_container, local_container)
    evaluator['eval_ng_zeroshot_recall'].prepare_zeroshot(global_container, local_container)

    if mode == 'predcls':
        local_container['pred_boxes'] = local_container['gt_boxes']
        local_container['pred_classes'] = local_container['gt_classes']
        local_container['obj_scores'] = np.ones(local_container['gt_classes'].shape[0])

    elif mode == 'sgcls':
        if local_container['gt_boxes'].shape[0] != local_container['pred_boxes'].shape[0]:
            # print('Num of GT boxes is not matching with num of pred boxes in SGCLS')
            pass
    elif mode == 'sgdet' or mode == 'phrdet':
        pass
    else:
        raise ValueError('invalid mode')
    """
    elif mode == 'preddet':
        # Only extract the indices that appear in GT
        prc = intersect_2d(pred_rel_inds, gt_rels[:, :2])
        if prc.size == 0:
            for k in result_dict[mode + '_recall']:
                result_dict[mode + '_recall'][k].append(0.0)
            return None, None, None
        pred_inds_per_gt = prc.argmax(0)
        pred_rel_inds = pred_rel_inds[pred_inds_per_gt]
        rel_scores = rel_scores[pred_inds_per_gt]
        # Now sort the matching ones
        rel_scores_sorted = argsort_desc(rel_scores[:,1:])
        rel_scores_sorted[:,1] += 1
        rel_scores_sorted = np.column_stack((pred_rel_inds[rel_scores_sorted[:,0]], rel_scores_sorted[:,1]))
        matches = intersect_2d(rel_scores_sorted, gt_rels)
        for k in result_dict[mode + '_recall']:
            rec_i = float(matches[:k].any(0).sum()) / float(gt_rels.shape[0])
            result_dict[mode + '_recall'][k].append(rec_i)
        return None, None, None
    """

    if local_container['pred_rel_inds'].shape[0] == 0:
        return

    # Traditional Metric with Graph Constraint
    # NOTE: this is the MAIN evaluation function, it must be run first (several important variables need to be update)
    local_container = evaluator['eval_recall'].calculate_recall(global_container, local_container, mode)

    # No Graph Constraint
    evaluator['eval_nog_recall'].calculate_recall(global_container, local_container, mode)
    # GT Pair Accuracy
    evaluator['eval_pair_accuracy'].calculate_recall(global_container, local_container, mode)
    # Mean Recall
    evaluator['eval_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # No Graph Constraint Mean Recall
    evaluator['eval_ng_mean_recall'].collect_mean_recall_items(global_container, local_container, mode)
    # Zero shot Recall
    evaluator['eval_zeroshot_recall'].calculate_recall(global_container, local_container, mode)
    # No Graph Constraint Zero-Shot Recall
    evaluator['eval_ng_zeroshot_recall'].calculate_recall(global_container, local_container, mode)

    return 



def convert_relation_matrix_to_triplets(relation):
    triplets = []
    for i in range(len(relation)):
        for j in range(len(relation)):
            if relation[i, j] > 0:
                triplets.append((i, j, relation[i, j]))
    return torch.LongTensor(triplets) # (num_rel, 3)


def generate_attributes_target(attributes, num_attributes):
        """
        from list of attribute indexs to [1,0,1,0,...,0,1] form
        """
        max_att = attributes.shape[1]
        num_obj = attributes.shape[0]

        with_attri_idx = (attributes.sum(-1) > 0).long()
        without_attri_idx = 1 - with_attri_idx
        num_pos = int(with_attri_idx.sum())
        num_neg = int(without_attri_idx.sum())
        assert num_pos + num_neg == num_obj

        attribute_targets = torch.zeros((num_obj, num_attributes), device=attributes.device).float()

        for idx in torch.nonzero(with_attri_idx).squeeze(1).tolist():
            for k in range(max_att):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1

        return attribute_targets