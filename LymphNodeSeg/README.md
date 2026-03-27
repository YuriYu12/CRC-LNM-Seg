# Lymph Node Segmentation
Generative-augmented and vascular-guided lymph node segmentation.

## Install
run `pip install -e .`

## Paths
```
export nnUNet_raw="/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="/data/nnunet/nnUNet_results"
```

## Commands
modify paths and run `python preprocess.py` to construct datasets (same as nnunetv2)

run `readme.py` to preprocess, train, and test using the following commands

### baseline
```
python readme.py --id 310 --mode dataset
python readme.py --id 310 --mode preprocess
python readme.py --id 310 --mode train
python readme.py --id 310 --mode test
```
### pre-train
```
python readme.py --id 311 --mode dataset
python readme.py --id 311 --mode preprocess --ref_id 310
python readme.py --id 311 --mode train
python readme.py --id 311 --mode test
```
### fine-tune
```
python readme.py --id 312 --mode train --config 3d_fullres_vesseldecattn --ref_id 311 --trainer nnUNetTrainer_oversample_ft_1en3
python readme.py --id 312 --mode test --config 3d_fullres_vesseldecattn --ref_id 311 --trainer nnUNetTrainer_oversample_ft_1en3
```