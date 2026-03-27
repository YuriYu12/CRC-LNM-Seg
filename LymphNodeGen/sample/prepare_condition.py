import os
import sys
import glob
import hydra
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
import scipy.ndimage as ndimage
from omegaconf import DictConfig, open_dict
from monai.transforms import Compose, RandAffined, Rand3DElasticd, RandCropByPosNegLabeld

sys.path.append(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.get_dataset import get_dataset


@hydra.main(config_path="../config", config_name="base_cfg", version_base=None)
def run(cfg: DictConfig):
    with open_dict(cfg):
        cfg.model.samples_folder = os.path.join(
            cfg.model.samples_folder, cfg.dataset.name, cfg.model.samples_folder_postfix
        )
    samples_folder_image = Path(os.path.join(cfg.model.samples_folder, "image"))
    samples_folder_image.mkdir(exist_ok=True, parents=True)
    samples_folder_mask = Path(os.path.join(cfg.model.samples_folder, "mask"))
    samples_folder_mask.mkdir(exist_ok=True, parents=True)
    samples_folder_supp = Path(os.path.join(cfg.model.samples_folder, "supp"))
    samples_folder_supp.mkdir(exist_ok=True, parents=True)

    _, _, sample_dataset, _, _ = get_dataset(cfg)
    print(f"{len(sample_dataset)} sampling cases.")

    files = sample_dataset.filenames
    div = cfg.model.samples // len(files)
    mod = cfg.model.samples % len(files)
    paths = []
    for k in range(div):
        paths += files
    paths += files[:mod]

    spatial_size = (cfg.model.diffusion_depth_size, cfg.model.diffusion_img_size, cfg.model.diffusion_img_size)
    INITIAL_TRANSFORMS = Compose(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label", "multi"], label_key="label", spatial_size=spatial_size, pos=1, neg=0
            )
        ]
    )

    CONVERT_TRANSFORMS = Compose(
        [
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

    for i in tqdm(range(cfg.model.samples), desc="preparing process"):
        print("idx:", i)
        image = np.load(paths[i])
        mask = np.load(paths[i].replace("image", "mask"))
        multi = np.load(paths[i].replace("image", "supp", 1).replace("image", "multi"))
        print(image.shape, mask.shape, multi.shape, image.dtype, mask.dtype, multi.dtype)

        data_dict = {"image": image[np.newaxis, :], "label": mask[np.newaxis, :], "multi": multi}
        data_dict = INITIAL_TRANSFORMS(data_dict)
        image = data_dict[0]["image"].squeeze(0).numpy()
        mask = data_dict[0]["label"].squeeze(0).numpy()
        multi = data_dict[0]["multi"].numpy()
        print(image.shape, mask.shape, multi.shape, image.dtype, mask.dtype, multi.dtype)

        nodule = np.where(mask)
        assert np.sum(multi[13][nodule] == 0) == 0

        annotation, count = ndimage.label(mask)
        assert count > 0
        if count > 1 and random.random() < 0.3:
            mask[np.where(annotation == random.randint(1, count))] = 0

        for times in range(21):
            if times == 20:
                multi = np.concatenate((mask[np.newaxis, :], multi[1:, ...]), axis=0).astype(np.uint8)
                np.save(os.path.join(samples_folder_image, "Cond_%s_%04d_image.npy" % (cfg.dataset.name, i + 1)), image)
                np.save(os.path.join(samples_folder_mask, "Cond_%s_%04d_mask.npy" % (cfg.dataset.name, i + 1)), mask)
                np.save(os.path.join(samples_folder_supp, "Cond_%s_%04d_multi.npy" % (cfg.dataset.name, i + 1)), multi)
                supp = np.zeros_like(mask).astype(np.uint8)
                for channel in range(multi.shape[0]):
                    supp[np.where(multi[channel] == 1)] = channel
                image = sitk.GetImageFromArray(image)
                mask = sitk.GetImageFromArray(mask)
                supp = sitk.GetImageFromArray(supp)
                sitk.WriteImage(
                    image, os.path.join(samples_folder_image, "Cond_%s_%04d_image.nii.gz" % (cfg.dataset.name, i + 1))
                )
                sitk.WriteImage(
                    mask, os.path.join(samples_folder_mask, "Cond_%s_%04d_mask.nii.gz" % (cfg.dataset.name, i + 1))
                )
                sitk.WriteImage(
                    supp, os.path.join(samples_folder_supp, "Cond_%s_%04d_multi.nii.gz" % (cfg.dataset.name, i + 1))
                )
                print(str(times) + " over")
            else:
                data_dict = {"image": image[np.newaxis, :], "label": mask[np.newaxis, :]}
                data_dict = CONVERT_TRANSFORMS(data_dict)
                img = data_dict["image"].squeeze(0).numpy()
                msk = data_dict["label"].squeeze(0).numpy().astype(np.int16)

                msk = ndimage.binary_closing(msk, structure=np.ones((2, 2, 2))).astype(np.int16)

                nodule = np.where(msk)
                print(np.sum(multi[13][nodule] == 0))
                if np.sum(msk) > 20 and np.sum(multi[13][nodule] == 0) < 80:
                    multi = np.concatenate((msk[np.newaxis, :], multi[1:, ...]), axis=0).astype(np.uint8)
                    np.save(
                        os.path.join(samples_folder_image, "Cond_%s_%04d_image.npy" % (cfg.dataset.name, i + 1)), img
                    )
                    np.save(os.path.join(samples_folder_mask, "Cond_%s_%04d_mask.npy" % (cfg.dataset.name, i + 1)), msk)
                    np.save(
                        os.path.join(samples_folder_supp, "Cond_%s_%04d_multi.npy" % (cfg.dataset.name, i + 1)), multi
                    )
                    supp = np.zeros_like(mask).astype(np.uint8)
                    for channel in range(multi.shape[0]):
                        supp[np.where(multi[channel] == 1)] = channel
                    image = sitk.GetImageFromArray(img)
                    mask = sitk.GetImageFromArray(msk)
                    supp = sitk.GetImageFromArray(supp)
                    sitk.WriteImage(
                        image,
                        os.path.join(samples_folder_image, "Cond_%s_%04d_image.nii.gz" % (cfg.dataset.name, i + 1)),
                    )
                    sitk.WriteImage(
                        mask, os.path.join(samples_folder_mask, "Cond_%s_%04d_mask.nii.gz" % (cfg.dataset.name, i + 1))
                    )
                    sitk.WriteImage(
                        supp, os.path.join(samples_folder_supp, "Cond_%s_%04d_multi.nii.gz" % (cfg.dataset.name, i + 1))
                    )
                    print(str(times) + " success")
                    break
                else:
                    print(str(times) + " ...")


if __name__ == "__main__":
    run()
