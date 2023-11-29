# Towards Fair Video Summarization

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Evaluation](#evaluation)
- [Training Models and Generating Summaries](#training-models-and-generating-summaries)

## Environment Setup

- Python 3.8.16
- CUDA 11.1
- Ubuntu 20.04.03

### Anaconda

``` conda env create -f environment.yml ```

## Dataset

All necessary data files are already provided in the repo. `datasets/fvs.h5` contains the *FairVidSum* dataset. `splits/fvs_splits.json` is the splits used for our benchmarks. `fair_npy_data/` contains all fairness labels and data required for SumBal evaluations and generating summaries using FVS-LP. `fair_npy_data/` can also be generated using `src/extract_fair_labels.py`

To generate `fair_npy_data/` yourself, please download [faces_fvs_tvsum.csv](https://www.dropbox.com/s/r93js11ifev964f/faces_fvs_tvsum.csv?dl=0), place it in `src/` and run ```python extract_fair_labels.py```. 

Please note the output of `extract_fair_labels.py` provides the dictionary to translate the int labels (for violating groups output from SumBal) to groups/sex/individuals. Our output is provided in `src/fair_labels_info.log`

## Evaluation

Please ensure `fair_npy_data/` is downloaded. All our generated model summaries are already provded in repo in respective models directories (`models/`). Please follow [Training Models and Generating Summaries](#training-models-and-generating-summaries) to train models and generate new model summaries.

### Average F1 Scores
To evaluate all models on all splits please run:
``` 
cd eval
python eval_fscore.py -m all
```

### SumBals
To evaluate all models on all splits please run:
``` 
cd eval
python eval_sumbal_all.py -m all
```

## Training Models and Generating Summaries
All (best) summaries from trained models are stored per split as `.pkl` files. Our generated summaries are already provided.

To train and generate summaries yourself, please follow the steps below for ever model.

`<split_idx>` is commonly used as an input to various training, which defines the split to train and generate summaries for and will be in range [0-4] for our 5 splits.

***Note:*** AC-SUM-GAN, CA-SUM, SUM-GAN-AAE, SUM-GAN-SL, PGL-SUM follow same procedure, where train and summaries are generated at same time using a single script for each individual split in similar fashion. Summaries for these models will be stores in `models/<model_name>/exp2/fvs/summaires`

### FVS-LP
```
cd lp-fvs
python lp-fvs.py
```
All generated summaries stored in `models/lp-fvs/summaries`

### Random
```
cd random
python gen_rand_summary.py
```
All generated summaries stored in `models/random/summaries`


### Human
```
cd human
python gen_user_summary.py
```
All generated summaries stored in `models/human/summaries`

### AC-SUM-GAN
```
cd models/ac-sum-gan
python ac-sum-gan.py --split_index <split_idx>
```
Please repeat and run for split_index=[0,1,2,3,4] to generate summaries for all splits.

Summaries are stored in `models/ac-sum-gan/exp2/fvs/summaires`, trained models (as .pth) and training fscore results (as .txt files) will appear in `models/ac-sum-gan/exp2/fvs/models` and `models/ac-sum-gan/exp2/fvs/results` respectively. All results are per split and will be in folders in format `split<split_idx>/`.

### SUM-GAN-AAE
Same procedure as AC-SUM-GAN. Run for all `split_idx=[0,1,2,3,4]`
```
cd models/sum-gan-aae
python sum-gan-aae.py --split_index <split_idx>
```

### SUM-GAN-SL
Same procedure as AC-SUM-GAN. Run for all `split_idx`
```
cd models/sum-gan-sl
python sum-gan-sl.py --split_index <split_idx>
```

### PGL-SUM
Same procedure as AC-SUM-GAN. Run for all `split_idx`
```
cd models/pgl-sum
python pgl-sum.py --split_index <split_idx>
```

### CA-SUM

Same procedure as AC-SUM-GAN. Run for all `split_idx`

```
cd models/ca-sum
python ca-sum.py --split_index <split_idx>
```


### DSNet
For DSNet, training and generating summaries are seperated. DSNet uses `fvs_splits.yml` instead of `fvs_splits.json`. The .yml splits file is already provided in `dsnet/splits/`

#### Train DSNet
DSNet trains for all splits. To train:
```
cd dsnet/src
python train.py anchor-free --model-dir models/fvs/ --splits ../splits/fvs.yml
```
All saved models (checkpoints) will be in `dsnet/src/models/fvs/checkpoint` for each split. Saved in format: `fvs.yml.<split_idx>.pt` for example `fvs.yml.0.pt`

#### Generate DSNet summaries
```
cd models/dsnet/src
python gen_summaries.py --ckpt_path <path_to_chkpt> –-split
<split_idx>
```

For instance, to generate summaries for Split 0:
```
python gen_summaries.py --ckpt_path models/fvs/checkpoint/fvs.yml.0.pt –-split 0
```

Please generate summaries for all 5 splits [0,1,2,3,4]. All generated summaries are stored in `dsnet/src/models/fvs/summaires`

### VASNet
Similar to DSNet, training and generating summaries are seperated and all splits will be trained with one script.

#### Train VasNet
```
cd models/vasnet/VASNet
python main.py -r ../VASNet -d ../../../datasets/fvs.h5
 -s ../../../splits/fvs_splits.json -t
```
All trained models will be asved in `vasnet/VASNet/data/models` in format `fvs_splits_<split_idx>_<best_fscore>.tar.pth`

#### Generate VASNet summaries
```
cd models/vasnet/VASNet
python gen_summaries.py --model_path <path_to_chkpt> --split <split_idx>
```
For instance, to generate summaries for Split 0:
```
python gen_summaries.py --model_path models/fvs/checkpoint/fvs_splits_0_0.6613551421287026.tar.pth –-split 0
```
Please generate summaries for all 5 splits [0,1,2,3,4]. All generated summaries are stored in `vasnet/VASNet/data/summaires`

### SUM-IND

Similar to AC-GAN-SUM like training procedures, SUM-IND script trains and generates summaries per split.

```
 python sum_ind/main.py -d ../../datasets/fvs.h5 -s ../../splits/fvs_splits.json -m tvsum --split-id <split_idx> --save-dir sum-ind<split_idx>
```

Please run for all 5 splits `split_idx=[0,1,2,3,4]`. Generated summaries are stored in `models/sum-ind/summaries`.
