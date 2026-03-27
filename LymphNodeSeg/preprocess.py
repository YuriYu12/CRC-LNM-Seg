import os
import json
import shutil
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt


def preprocess(image_path, label_path, folder_path, vessel_path=None):
    os.makedirs(os.path.join(folder_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "labelsTs"), exist_ok=True)

    with open("data_split.json", "r") as file:
        data = json.load(file)["idx"]

    for file in sorted(os.listdir(image_path)):
        print(file)
        image = sitk.ReadImage(os.path.join(image_path, file))
        image_array = sitk.GetArrayFromImage(image)
        label = sitk.ReadImage(os.path.join(label_path, file.replace("image", "mask")))
        label_array = sitk.GetArrayFromImage(label)
        segmentation = sitk.ReadImage(os.path.join(image_path, file).replace("image", "segmentation"))
        segmentation_array = sitk.GetArrayFromImage(segmentation)
        origin = image.GetOrigin()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        print(spacing)
        print(image_array.shape, label_array.shape)

        label_array[label_array > 1] = 1

        if vessel_path is not None:
            vessel = sitk.ReadImage(os.path.join(vessel_path, file.replace("image", "vessel")))
            vessel_array = sitk.GetArrayFromImage(vessel)
            vessel_map = np.zeros_like(vessel_array)
            vessel_map[vessel_array > 0] = 1
            distance_array = distance_transform_edt(1 - vessel_map, sampling=vessel.GetSpacing()[::-1])
            distance_array = distance_array.astype(np.float32)

        idx = int(file[4:7])
        if idx in data["private_ts_idx"]:
            bound = 5
            region = np.where(segmentation_array == 20)  # colon region (totalsegmentator v2)
            pre_image_path = os.path.join(folder_path, "imagesTs")
            pre_label_path = os.path.join(folder_path, "labelsTs")
        else:
            bound = 100
            region = np.where(label_array == 1)  # nodule region
            pre_image_path = os.path.join(folder_path, "imagesTr")
            pre_label_path = os.path.join(folder_path, "labelsTr")

        boundz = round(bound / spacing[2])
        boundy = round(bound / spacing[1])
        boundx = round(bound / spacing[0])

        image_region = image_array[
            np.max([0, np.min(region[0]) - boundz]) : np.min([np.max(region[0]) + boundz + 1, image_array.shape[0]]),
            np.max([0, np.min(region[1]) - boundy]) : np.min([np.max(region[1]) + boundy + 1, image_array.shape[1]]),
            np.max([0, np.min(region[2]) - boundx]) : np.min([np.max(region[2]) + boundx + 1, image_array.shape[2]]),
        ]
        label_region = label_array[
            np.max([0, np.min(region[0]) - boundz]) : np.min([np.max(region[0]) + boundz + 1, label_array.shape[0]]),
            np.max([0, np.min(region[1]) - boundy]) : np.min([np.max(region[1]) + boundy + 1, label_array.shape[1]]),
            np.max([0, np.min(region[2]) - boundx]) : np.min([np.max(region[2]) + boundx + 1, label_array.shape[2]]),
        ]
        print(image_region.shape, label_region.shape)

        pre_image = sitk.GetImageFromArray(image_region)
        pre_image.SetOrigin(origin)
        pre_image.SetDirection(direction)
        pre_image.SetSpacing(spacing)
        sitk.WriteImage(pre_image, os.path.join(pre_image_path, "LNM_LYMPH_%03d_0000.nii.gz" % idx))
        pre_label = sitk.GetImageFromArray(label_region)
        pre_label.SetOrigin(origin)
        pre_label.SetDirection(direction)
        pre_label.SetSpacing(spacing)
        sitk.WriteImage(pre_label, os.path.join(pre_label_path, "LNM_LYMPH_%03d.nii.gz" % idx))

        if vessel_path is not None:
            vessel_region = vessel_array[
                np.max([0, np.min(region[0]) - boundz]) : np.min(
                    [np.max(region[0]) + boundz + 1, label_array.shape[0]]
                ),
                np.max([0, np.min(region[1]) - boundy]) : np.min(
                    [np.max(region[1]) + boundy + 1, label_array.shape[1]]
                ),
                np.max([0, np.min(region[2]) - boundx]) : np.min(
                    [np.max(region[2]) + boundx + 1, label_array.shape[2]]
                ),
            ]
            distance_region = distance_array[
                np.max([0, np.min(region[0]) - boundz]) : np.min(
                    [np.max(region[0]) + boundz + 1, label_array.shape[0]]
                ),
                np.max([0, np.min(region[1]) - boundy]) : np.min(
                    [np.max(region[1]) + boundy + 1, label_array.shape[1]]
                ),
                np.max([0, np.min(region[2]) - boundx]) : np.min(
                    [np.max(region[2]) + boundx + 1, label_array.shape[2]]
                ),
            ]
            pre_vessel = sitk.GetImageFromArray(vessel_region)
            pre_vessel.SetOrigin(origin)
            pre_vessel.SetDirection(direction)
            pre_vessel.SetSpacing(spacing)
            sitk.WriteImage(pre_vessel, os.path.join(pre_image_path, "LNM_LYMPH_%03d_0001.nii.gz" % idx))
            pre_distance = sitk.GetImageFromArray(distance_region)
            pre_distance.SetOrigin(origin)
            pre_distance.SetDirection(direction)
            pre_distance.SetSpacing(spacing)
            sitk.WriteImage(pre_distance, os.path.join(pre_image_path, "LNM_LYMPH_%03d_0002.nii.gz" % idx))


def process(image_path, label_path, folder_path, ref_folder_path, num=1000):
    os.makedirs(os.path.join(folder_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "labelsTr"), exist_ok=True)
    shutil.copytree(os.path.join(ref_folder_path, "imagesTs"), os.path.join(folder_path, "imagesTs"))
    shutil.copytree(os.path.join(ref_folder_path, "labelsTs"), os.path.join(folder_path, "labelsTs"))

    for i in range(num):
        shutil.copyfile(
            os.path.join(image_path, "Syn_LNM_%04d_image.nii.gz" % (i + 1)),
            os.path.join(folder_path, "imagesTr", "LNM_LYMPH_%03d_0000.nii.gz" % (i + 139)),
        )
        shutil.copyfile(
            os.path.join(label_path, "Syn_LNM_%04d_mask.nii.gz" % (i + 1)),
            os.path.join(folder_path, "labelsTr", "LNM_LYMPH_%03d.nii.gz" % (i + 139)),
        )

    with open("data_split.json", "r") as file:
        data = json.load(file)["idx"]

    for idx in data["private_va_idx"]:
        shutil.copyfile(
            os.path.join(ref_folder_path, "imagesTr", "LNM_LYMPH_%03d_0000.nii.gz" % idx),
            os.path.join(folder_path, "imagesTr", "LNM_LYMPH_%03d_0000.nii.gz" % idx),
        )
        shutil.copyfile(
            os.path.join(ref_folder_path, "labelsTr", "LNM_LYMPH_%03d.nii.gz" % idx),
            os.path.join(folder_path, "labelsTr", "LNM_LYMPH_%03d.nii.gz" % idx),
        )

if __name__ == "__main__":
    # baseline
    preprocess(
        image_path="/data/LNM/image",
        label_path="/data/LNM/mask",
        folder_path="/data/nnunet/nnUNet_raw/Dataset310_LNM_LYMPH",
    )
    # pre-train
    process(
        image_path="sample/LNM/normal/image",
        label_path="sample/LNM/normal/mask",
        folder_path="/data/nnunet/nnUNet_raw/Dataset311_LNM_LYMPH",
        ref_folder_path="/data/nnunet/nnUNet_raw/Dataset310_LNM_LYMPH",
        num=1000,
    )
    # fine-tune
    preprocess(
        image_path="/data/LNM/image",
        label_path="/data/LNM/mask",
        vessel_path="/data/LNM/vessel",
        folder_path="/data/nnunet/nnUNet_raw/Dataset312_LNM_LYMPH",
    )