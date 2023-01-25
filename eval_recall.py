import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from data.mm_data.sg_raw import load_json
from data.mm_data.sgg_VG_dataset import SggVGDataset, VGDatasetReader
import sacrebleu
from sacrebleu.metrics import BLEU
from itertools import zip_longest
import pdb
from PIL import Image

# y_true is the array of true labels for the test data
# y_pred is the array of predicted labels for the test data

EVAL_BLEU_ORDER = 4

def calculate(y_true, y_pred):

    # # Create label encoder
    # encoder = LabelEncoder()

    # # Fit and transform y_true and y_pred
    # y_true_encoded = encoder.fit_transform(y_true)
    # y_pred_encoded = encoder.transform(y_pred)

    # print("Encoded y_true:", y_true_encoded)
    # print("Encoded y_pred:", y_pred_encoded)

    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for pred_token in y_pred:
        if pred_token in y_true:
            true_positive += 1
        else:
            false_positive += 1
    
    for true_token in y_true:
        if true_token in y_pred:
            continue
        else:
            false_negative += 1

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if true_positive + false_positive == 0:
        precision = 1
    else:
        precision = true_positive / (true_positive + false_positive)
    
    if true_positive + false_negative == 0:
        recall = 1
    else: 
        recall = true_positive / (true_positive + false_negative)
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

    # print('Accuracy:', accuracy)
    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('F1 Score:', f1)

    return accuracy, precision, recall, f1


def eval_bleu(hyps, refs):

    bleu_counts = []
    bleu_totals = []
    try: 
        bleu = sacrebleu.corpus_bleu(hyps, list(zip_longest(*refs)))
        # logging_output["_bleu_sys_len"] = bleu.sys_len
        # logging_output["_bleu_ref_len"] = bleu.ref_len
        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        assert len(bleu.counts) == EVAL_BLEU_ORDER
        for i in range(EVAL_BLEU_ORDER):
            bleu_counts.append(bleu.counts[i])
            bleu_totals.append(bleu.totals[i])
        
        # print("bleu counts: ", bleu_counts)
        # print("bleu totals: ", bleu_totals)
        return bleu_totals[0]
    except:
        bleu_score = 0.0
        return bleu_score


if __name__ == "__main__":
    
    predictions = load_json('results/sgg/VG/test_0115_pretrain_noorder_complete_predict.json')

    img_dir = '/data/c/zhuowan/gqa/data/images/'
    base_dir = 'dataset/sgg_data/VG'
    roidb_file = f'{base_dir}/VG-SGG-with-attri.h5'
    dict_file = f'{base_dir}/VG-SGG-dicts-with-attri.json'
    image_file = f'{base_dir}/image_data.json'
    tgt_seq_len = 350
    split = 'test'
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

    # groundtruths = []
    # predictions = dict()

    

    bleu_scores = []
    accuracy_labels = []
    precision_labels = []
    recall_labels = []
    f1_labels = []
    accuracy_rels = []
    precision_rels = []
    recall_rels = []
    f1_rels = []

            


    i = 0
    for pred in predictions:
        pred_label = []
        pred_relation = []
        gt_label = []
        gt_relation = []
        i += 1
        print(f'{i}/{len(predictions)}    0115_complete')
        for result_id, prediction in pred.items():
            image_idx = int(result_id.split('_')[1])
            sample_id = result_id.split('_')[0]

            img = Image.open(dataset.filenames[image_idx]).convert("RGB")
            target = dataset.get_groundtruth(image_idx)
            target_seq, obj_labels, rel_labels = dataset.target2seq_raw(img, target, required_len=tgt_seq_len, obj_order=False)

            img_name = sample_id + ".jpg"
            image_path = img_dir + img_name
            # print("image path: ", image_path)
            
            # Extract the bounding box separately




            # print("gt: ", target_seq)
            # print("pred: ", prediction)
            target_sentence = ' '.join(target_seq)
            bleu_score = eval_bleu(prediction, target_sentence)
            bleu_scores.append(bleu_score)

            pred_sentences = prediction.split('.')
            for pred_sent in pred_sentences:
                pred_tokens = pred_sent.split(' ')
                # print("pred tokens origin: ", pred_tokens)
                while '' in pred_tokens:
                    pred_tokens.remove('')
                
                # j = 0
                for j, pred_token in enumerate(pred_tokens):
                    if j == 0 or j == len(pred_tokens)-1:
                        pred_label.append(pred_tokens[j])
                        # j += 1
                    elif j < len(pred_tokens)-1 and ',' in pred_tokens[j+1]:
                        pred_label.append(pred_tokens[j])
                        # j += 1
                    # elif j < len(pred_tokens)-1 and pred_tokens[j+1] == '.':
                    #     pred_label.append(pred_token)
                    elif 'is' in pred_tokens[j] or ',' in pred_tokens[j]:
                        # j += 1
                        continue
                    else:
                        r_raw_list = []
                        # print("pred_tokens[j]: ", pred_tokens[j])
                        # print("pred_tokens[j-1]: ", pred_tokens[j-1])
                        if 'is' in pred_tokens[j-1] or ',' in pred_tokens[j-1]:
                            if j < (len(pred_tokens)-2): 
                                # print("j: ", j)
                                # print("pred_tokens[j] origin: ", pred_tokens[j])
                                try:
                                    while (',' not in pred_tokens[j+1]):   # while j<len(pred_tokens)-1:
                                        r_raw_list.append(pred_tokens[j])
                                        j += 1
                                        if j >= len(pred_tokens)-1:
                                            break
                                except:
                                    print("pred tokens: ", pred_tokens)
                                    print("pred_tokens[j]: ", pred_tokens[j])
                                    print("pred_tokens[j+1]: ", pred_tokens[j+1])
                                    print("pred_tokens[j+2]: ", pred_tokens[j+2])
                            else:
                                while j<len(pred_tokens)-1: #  and j<len(pred_tokens)-2
                                    # print("j: ", j)
                                    r_raw_list.append(pred_tokens[j])
                                    j += 1
                            r_raw = ' '.join(r_raw_list)
                            pred_relation.append(r_raw)
                        else:
                            # j += 1
                            continue
            
   
            gt_label = obj_labels
            gt_relation = rel_labels

            # pred_label.sort()
            # pred_relation.sort()
            # gt_label.sort()
            # gt_relation.sort()

            # if len(pred_label) != len(gt_label):
            #     min_label_len = min(len(pred_label), len(gt_label))
            #     pred_label = pred_label[:min_label_len]
            #     gt_label = gt_label[:min_label_len]
            
            # if len(pred_relation) != len(gt_relation):
            #     min_relation_len = min(len(pred_relation), len(gt_relation))
            #     pred_relation = pred_relation[:min_relation_len]
            #     gt_relation = gt_relation[:min_relation_len]

            # print("pred_label: ", pred_label)
            # print("pred_relation: ", pred_relation)
            # print("gt_label: ", gt_label)
            # print("gt_relation: ", gt_relation)

            

            
            accuracy_label, precision_label, recall_label, f1_label = calculate(y_true=gt_label, y_pred=pred_label)
            
            accuracy_rel, precision_rel, recall_rel, f1_rel = calculate(y_true=gt_relation, y_pred=pred_relation)

            accuracy_labels.append(accuracy_label)
            precision_labels.append(precision_label)
            recall_labels.append(recall_label)
            f1_labels.append(f1_label)
            accuracy_rels.append(accuracy_rel)
            precision_rels.append(precision_rel)
            recall_rels.append(recall_rel)
            f1_rels.append(f1_rel)

            # pdb.set_trace()
                    


    print("average bleu scores: ", sum(bleu_scores)/len(bleu_scores))

    print("\naverage scores of object labels: ")
    print("accuracy: ", sum(accuracy_labels)/len(accuracy_labels))
    print("precision: ", sum(precision_labels)/len(precision_labels))
    print("recall: ", sum(recall_labels)/len(recall_labels))
    print("f1: ", sum(f1_labels)/len(f1_labels))

    print("\naverage scores of relations: ")
    print("accuracy: ", sum(accuracy_rels)/len(accuracy_rels))
    print("precision: ", sum(precision_rels)/len(precision_rels))
    print("recall: ", sum(recall_rels)/len(recall_rels))
    print("f1: ", sum(f1_rels)/len(f1_rels))
