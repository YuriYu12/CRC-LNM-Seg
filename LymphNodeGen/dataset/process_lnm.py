import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from skimage import measure


def get_real_resize_factor(spacing, new_spacing, image_array):
    resize_factor = spacing / new_spacing
    new_real_shape = image_array.shape * resize_factor[::-1]
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image_array.shape
    return real_resize_factor


def extract_air_and_body(image_array):
    binary_array = image_array.copy()
    binary_array[image_array >= -500] = 1
    binary_array[image_array <= -500] = 0
    binary = sitk.GetImageFromArray(binary_array)

    ConnectedThresholdImageFilter = sitk.ConnectedThresholdImageFilter()
    ConnectedThresholdImageFilter.SetLower(0)
    ConnectedThresholdImageFilter.SetUpper(0)
    ConnectedThresholdImageFilter.SetSeedList(
        [(0, 0, 0), (image_array.shape[2] - 1, image_array.shape[1] - 1, image_array.shape[0] - 1)]
    )

    air = ConnectedThresholdImageFilter.Execute(binary)
    air = sitk.BinaryMorphologicalClosing(air)
    body = sitk.ShiftScale(air, -1, -1)
    body_array = sitk.GetArrayFromImage(body)

    # maximum connected component
    body_array = measure.label(body_array, connectivity=3)
    props = measure.regionprops(body_array)
    largest = max(props, key=lambda x: x.area).label
    body_array = (body_array == largest).astype(np.uint8)
    body_array = sitk.GetArrayFromImage(sitk.BinaryMorphologicalOpening(sitk.GetImageFromArray(body_array)))
    return body_array.astype(np.uint8)


def get_crop_bbox(shape, region, bound, region_size):
    Z, Y, X = shape

    z_min, z_max = region[0].min(), region[0].max()
    y_min, y_max = region[1].min(), region[1].max()
    x_min, x_max = region[2].min(), region[2].max()

    z0 = max(0, z_min - bound)
    z1 = min(Z, z_max + bound + 1)
    y0 = max(0, y_min - bound)
    y1 = min(Y, y_max + bound + 1)
    x0 = max(0, x_min - bound)
    x1 = min(X, x_max + bound + 1)

    dz = max(0, region_size[0] - (z1 - z0))
    dy = max(0, region_size[1] - (y1 - y0))
    dx = max(0, region_size[2] - (x1 - x0))

    if dz + dy + dx > 0:
        z0 = max(0, z_min - bound - dz)
        z1 = min(Z, z_max + bound + dz + 1)
        y0 = max(0, y_min - bound - dy)
        y1 = min(Y, y_max + bound + dy + 1)
        x0 = max(0, x_min - bound - dx)
        x1 = min(X, x_max + bound + dx + 1)

    return z0, z1, y0, y1, x0, x1


def nodule_process(image_path, mask_path, segmentation_path, process_path):
    for file in sorted(os.listdir(image_path)):
        print(file)
        image = sitk.ReadImage(os.path.join(image_path, file), sitk.sitkFloat32)
        image_array = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(os.path.join(mask_path, file.replace("image", "mask")), sitk.sitkInt16)
        mask_array = sitk.GetArrayFromImage(mask)
        ml = sitk.ReadImage(os.path.join(segmentation_path, file.replace("image", "segmentation")))
        ml_array = sitk.GetArrayFromImage(ml)
        spacing = image.GetSpacing()
        print("original")
        print(spacing)
        print(image_array.shape)

        mask_array[mask_array > 1] = 1

        if not np.any(mask_array):
            continue

        # air and body
        body_array = extract_air_and_body(image_array)
        # totalsegmentator v2
        ml_array[np.where((ml_array == 0) & (body_array == 1))] = 118
        ml_sep_array = ml_array.copy()
        ml_array[np.where(mask_array)] = 119
        ml_sep_array[np.where(mask_array)] = 118

        print("resample")
        new_spacing = [1.0, 1.0, 1.0]
        real_resize_factor = get_real_resize_factor(np.array(spacing), new_spacing, image_array)
        image_array = ndimage.zoom(image_array, real_resize_factor, order=3)
        mask_array = ndimage.zoom(mask_array, real_resize_factor, order=0)
        ml_array = ndimage.zoom(ml_array, real_resize_factor, order=0)
        ml_sep_array = ndimage.zoom(ml_sep_array, real_resize_factor, order=0)
        print(image_array.shape)

        print("nodule region")
        region_size = (128, 128, 128)
        bound = 100
        region = np.where(mask_array == 1)  # nodule region
        z_min, z_max, y_min, y_max, x_min, x_max = get_crop_bbox(image_array.shape, region, bound, region_size)
        image_region = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
        mask_region = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]
        ml_region = ml_array[z_min:z_max, y_min:y_max, x_min:x_max]
        ml_sep_region = ml_sep_array[z_min:z_max, y_min:y_max, x_min:x_max]
        print(image_region.shape)

        print("normalize")
        image_region = np.clip(image_region, -120, 240)
        image_region = (image_region - 60) / 180
        print(np.min(image_region), np.max(image_region))
        print(image_region.dtype, mask_region.dtype)
        print(ml_region.dtype, ml_sep_region.dtype)

        if not np.any(mask_region):
            continue

        image = sitk.GetImageFromArray(image_region)
        image.SetSpacing(new_spacing)
        sitk.WriteImage(image, os.path.join(os.path.join(process_path, "image"), file))
        np.save(os.path.join(os.path.join(process_path, "image"), file[:-7] + ".npy"), image_region)

        mask = sitk.GetImageFromArray(mask_region)
        mask.SetSpacing(new_spacing)
        sitk.WriteImage(mask, os.path.join(os.path.join(process_path, "mask"), file.replace("image", "mask")))
        np.save(
            os.path.join(os.path.join(process_path, "mask"), file[:-7].replace("image", "mask") + ".npy"),
            mask_region,
        )

        ml = sitk.GetImageFromArray(ml_region)
        ml.SetSpacing(new_spacing)
        sitk.WriteImage(ml, os.path.join(os.path.join(process_path, "supp"), file.replace("image", "ml")))
        np.save(
            os.path.join(os.path.join(process_path, "supp"), file[:-7].replace("image", "ml") + ".npy"),
            ml_region,
        )

        ml_sep = sitk.GetImageFromArray(ml_sep_region)
        ml_sep.SetSpacing(new_spacing)
        sitk.WriteImage(ml_sep, os.path.join(os.path.join(process_path, "supp"), file.replace("image", "ml_sep")))
        np.save(
            os.path.join(os.path.join(process_path, "supp"), file[:-7].replace("image", "ml_sep") + ".npy"),
            ml_sep_region,
        )

        from totalseg_classmap import convert_map

        multi_backup_region = np.zeros_like(ml_region, dtype=np.uint8)
        for new_label, old_labels in convert_map["total_v2"].items():
            multi_backup_region[np.isin(ml_region, old_labels)] = new_label
        print(multi_backup_region.shape)

        channel = 15
        multi_region = np.zeros((channel, *mask_region.shape), dtype=np.uint8)
        multi_region[0] = mask_region == 1
        multi_region[1:] = multi_backup_region[None] == np.arange(1, channel)[:, None, None, None]
        print(multi_region.shape)

        multi = sitk.GetImageFromArray(multi_backup_region)
        multi.SetSpacing(new_spacing)
        sitk.WriteImage(multi, os.path.join(os.path.join(process_path, "supp"), file.replace("image", "multi")))
        np.save(
            os.path.join(os.path.join(process_path, "supp"), file.replace("image", "multi").replace("nii.gz", "npy")),
            multi_region,
        )


if __name__ == "__main__":
    image_path = "/data/LNM/image"
    mask_path = "/data/LNM/mask"
    segmentation_path = "/data/LNM/segmentation"

    process_path = "/data/LNM/process"
    os.makedirs(os.path.join(process_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(process_path, "mask"), exist_ok=True)
    os.makedirs(os.path.join(process_path, "supp"), exist_ok=True)
    nodule_process(image_path, mask_path, segmentation_path, process_path)
