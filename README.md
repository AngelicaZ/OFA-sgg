**OFA for scene graph generation (SGG)**

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

