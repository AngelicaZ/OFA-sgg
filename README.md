**OFA for scene graph generation (SGG)**

The model is based on OFA(https://github.com/OFA-Sys/OFA), which is a unified multimodal pretrained Transformer model that unifies modalities. In this task, we propose integrating the SGG task into a unified model with other vision-language tasks. Our approach
involves representing the scene graph as a sequence of objects, bounding boxes and relationships, and using a sequence-to-sequence (Seq2Seq) pipeline to generate an output sequence that is then converted back into the form of a scene graph. This allows the SGG task to be treated as a Seq2Seq task within the same unified model as various other multimodal tasks.


<br></br>

## Dataset Preparation
### GQA
Download the GQA dataset: train_sceneGraphs.json and val_sceneGraphs.json  
attrlabel_glove_taxo.npy, sgg_features.h5, sgg_info.json, gqa_vocab_taxo.json, new_vocab_0822.json  

### VG
Download the VG dataset: VG-SGG-with-attri.h5, VG-SGG-dicts-with-attri.json, image_data.json  
Download the [VG dataset h5 file](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed)

<br></br>

## Train and Test
Please modify the paths of datasets in the sh files.

To train the SGG task for GQA dataset, run 
```
cd run_sripts/sgg
sh train_sgg_GQA.sh
```
For VG dataset, run
```
sh train_sgg_VG.sh
```

To test the SGG task for GQA or VG dataset, run `sh eval_sgg_GQA.sh` or `eval_sgg_VG.sh`

## Model Structure
### Task Overview
![task_overview](pictures/task_overview_png.png)

### Scene Graph Generation
![SGG](pictures/sgg_png.png)


## Experiments and Results
### SGdet
| Models   | mAp  | R@20  | R@50  | R@100 | ng-R@20 | ng-R@50 |ng-R@100 | zR@20 | zR@50 | zR@100 | mR@20 | mR@50 | mR@100 |
|----------|------|-------|-------|-------|---------|---------|---------|-------|-------|--------|-------|-------|--------|
| VCTree   | --   | 24.53 | 31.93 | 36.21 | 26.14   | 35.73   | 42.34   | 0.1   | 0.31  | 0.69   | 5.38  | 7.44  | 8.66   |
| Ofa_tiny | 1.16 | 0.28  | 0.28  | 0.28  | 0.28    | 0.28    | 0.28    | 0.01  | 0.01  | 0.01   | 0.08  | 0.08  | 0.08   |

### SGCls

### PredCls
| Models | mAp | R@20 | R@50 | R@100 | ng-R@20 | ng-R@50 |ng-R@100 | zR@20 | zR@50 | zR@100 | mR@20 | mR@50 | mR@100 |  
|--------|-----|------|------|-------|---------|---------|---------|-------|-------|--------|-------|-------|--------|  
| VCTree | -- | 59.02 | 65.42 | 67.18 | 67.2 | 81.63 | 88.83 | 1.04 | 3.27 | 5.51 | 13.12 | 16.74 | 18.16 |
| Ofa_tiny | 1.75 | 0.08 | 0.08 | 0.08 | 1.44 | 2.84 | 3.82 | 0.23 | 0.23 | 0.23 | 0.09 | 0.09 | 0.09 |

### Overfit
Overfit

overfit with objnum loss

Overfit with bbox loss and objnum loss

### Aligh the number of prediction bbox to ground truth
set the labels to ground truth, and append the ground truth bbox to the prediction to match the label number, the result should be good.

Test on 0203_PredCls


## Visualization
Preciction: ['cat', [0, 62, 498, 242], 'is', 'on', 'bed', [0, 0, 498, 373], ',', 'has', 'ear', [417, 96, 52, 60], ',earing', 'pant', [0, 172, 230, 141], ',<bin_0><bin_0><bin_996>', '.']  
Groundtruth: ['cat ', [9], [58], [483], [305], ' is ', 'in ', 'chair ', [47], [274], [203], [368], ' . ']  
![visualization1](pictures/visualization1.png)

Preciction: ['man', [102, 36, 86, 117], 'is', 'wearing', 'jean', [138, 80, 33, 63], ',', 'on', 'skateboard', [129, 135, 63, 24], '.']  
Groundtruth: ['man ', [102], [36], [188], [153], ' is ', 'has ', 'leg ', [159], [78], [188], [144], ' , ', 'wears ', 'pant ', [138], [75], [188], [144], ' , ', 'riding ', 'skateboard ', [129], [135], [192], [160], ' . ', 'man ', [104], [37], [191], [147], ' is ', 'has ', 'leg ', [138], [80], [171], [144], ' . ']  
![visualization2](pictures/visualization2.png)


