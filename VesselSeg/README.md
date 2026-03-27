# Vessel Segmentation
Vessel segmentation training and inference.


## preprocess
`python train.py --preprocess --name normalloss --preprocess_dir <path_to_the_dataset>`

## pre-train
`python train.py --train --no_sbatch --desc all --name normalloss`

## fine-tune
`python train.py --train --no_sbatch --desc all_ft0/all_ft1/all_ft2/all_ft3/all_ft4 --name normalloss`

## inference
```
python seg.py --gpu 0 1 2 3 4 \
    --image_dir /data/LNM/image/ \
    --totalseg_dir /data/LNM/segmentation \
    --save_dir /data/LNM/vessel \
    --nfile_per_iter 1
```