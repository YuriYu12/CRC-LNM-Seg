import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, RandAffined, Rand3DElasticd, RandCropByPosNegLabeld


def get_transforms(size, multi):
    if multi:
        keys = ["image", "label", "multi"]
    else:
        keys = ["image", "label"]

    TRANSFORMS = RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=size, pos=1, neg=0)
    CONVERT_TRANSFORMS = Compose(
        [
            RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=size, pos=1, neg=0),
            RandAffined(
                keys=["image", "label"],
                prob=0.8,
                rotate_range=(0.4, 0.4, 0.4),
                shear_range=(0.2, 0.2, 0.2),
                scale_range=(0.2, 0.2, 0.2),
                translate_range=(20, 20, 20),
                mode=["bilinear", "nearest"],
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.8,
                sigma_range=(5, 7),
                magnitude_range=(40, 80),
                mode=["bilinear", "nearest"],
            ),
        ]
    )
    return TRANSFORMS, CONVERT_TRANSFORMS


class LNMDataset(Dataset):
    def __init__(self, root_dir="", image_size=128, depth_size=128, multi=False, flag=""):
        self.flag = flag
        self.multi = multi
        self.transforms, self.convert_transforms = get_transforms((depth_size, image_size, image_size), multi)
        self.filenames = sorted(glob.glob(os.path.join(root_dir, "*.npy")))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        path = self.filenames[index]
        img = np.load(path)
        msk = np.load(path.replace("image", "mask"))
        img = img[np.newaxis, :]
        msk = msk[np.newaxis, :]

        if self.multi:
            multi = np.load(path.replace("image", "supp", 1).replace("image", "multi"))
            data_dict = {"image": img, "label": msk, "multi": multi}
            if self.flag == "sample":
                data_aug = self.convert_transforms(data_dict)
            else:
                data_aug = self.transforms(data_dict)
            img = data_aug[0]["image"]
            msk = data_aug[0]["label"]
            multi = data_aug[0]["multi"]
            return {"data": img.float(), "mask": msk.float(), "multi": multi.float()}
        else:
            data_dict = {"image": img, "label": msk}
            if self.flag == "sample":
                data_aug = self.convert_transforms(data_dict)
            else:
                data_aug = self.transforms(data_dict)
            img = data_aug[0]["image"]
            msk = data_aug[0]["label"]
            return {"data": img.float(), "mask": msk.float()}
