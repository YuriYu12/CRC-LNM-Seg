# Lymph Node Generation
The conditional diffusion model for lymph node synthesis.

## Data
data structure
```
/data/LNM/
├── image/
│   └── LNM_001_image.nii.gz
├── mask/
│   └── LNM_001_mask.nii.gz
└── segmentation/
    └── LNM_001_segmentation.nii.gz (obtained from TotalSegmentator v2)
```

data processing (modify paths and run `python dataset/process_lnm.py`)
```
/data/LNM/process/
├── image/
│   └── LNM_001_image.npy
├── mask/
│   └── LNM_001_mask.npy
└── supp/
    └── LNM_001_multi.npy
```

## Config
modify dataset config and model config in the `config` folder.

## DDPM Training
```
CUDA_VISIBLE_DEVICES=0 python train/train_ddpm.py model=ddpm dataset=lnm model.use_multimask=True model.use_control_mask=True model.use_spatial_transformer=True model.context_dim=256
```

## DDPM Sampling
```
python sample/prepare_condition.py model=ddpm dataset=lnm model.samples=1000 model.samples_folder_postfix='prepare'
```
```
CUDA_VISIBLE_DEVICES=0 python sample/sample_condition.py model=ddpm dataset=lnm dataset.root_dir='sample/LNM/prepare/image/' model.use_multimask=True model.use_control_mask=True model.use_spatial_transformer=True model.context_dim=256 model.samples=1000 model.ddpm_ckpt='model/LNM/ddpm/normal/model-100.pt'
```