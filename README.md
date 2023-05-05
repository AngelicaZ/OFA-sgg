**OFA for scene graph generation (SGG)**

The model is based on OFA(https://github.com/OFA-Sys/OFA), which is a unified multimodal pretrained Transformer model that unifies modalities.

<br></br>

## Dataset preparation
Download the [VG dataset h5 file](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed)
<br></br>

## Train and test
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
![task_overview](pictures/task_overview.pdf)

### Scene Graph Generation
![SGG](pictures/SGG.pdf)


## Experiments and Results
### SGdet


### PredCls

### Overfit

### 


## Visualization



