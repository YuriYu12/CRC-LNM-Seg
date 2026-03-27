import re, os, sys
from os.path import *
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
import shutil
import SimpleITK as sitk

imagepath = join(os.environ.get("nnUNet_raw"), "Dataset000CMU", "imagesTr")
savepath = join(os.environ.get("nnUNet_raw"), "Dataset010DroppedOrgans", "imagesTr")
totalsegpath = "/data/dataset/cmu/v2/mask"


def listdir(path):
    return [join(path, item) for item in os.listdir(path)]


def makedir_or_dirs(path, destory_on_exist=False):
    if destory_on_exist and exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def find_cor_label():
    refs = [_ for _ in listdir(imagepath) if _.split('/')[-1][len("Dataset000CMU_")] == '9']
    ref_sizes = [sitk.ReadImage(_).GetSize() for _ in refs]
    print(ref_sizes)
    ref_labels = [_ for _ in listdir(totalsegpath) if _.split('/')[-1][len("Dataset000CMU_")] == '9']
    label_sizes = [sitk.ReadImage(_).GetSize() for _ in ref_labels]
    print(label_sizes)
    
    for i, i_size in zip(refs, ref_sizes):
        index = 0
        while label_sizes[index] != i_size: index += 1
        os.rename(ref_labels[index], join(totalsegpath, i.split('/')[-1]))


def main(mean=-130, std=60):
    makedir_or_dirs(savepath, destory_on_exist=True)
    makedir_or_dirs(savepath.replace("imagesTr", "labelsTr"), destory_on_exist=True)
    for item in tqdm(sorted(listdir(imagepath))):
        imname = item.split('/')[-1]
        imid = re.findall("\d+", imname)[-2]
        maskname = join(imagepath.replace("imagesTr", "labelsTr"), imname.replace("_0000.nii.gz", ".nii.gz"))
        
        totalsegid = f"Dataset000CMU_{imid}"
        totalsegname = join(totalsegpath, imname)
        print(', '.join([imname, totalsegname, maskname]))
        
        raw = sitk.ReadImage(item)
        image = sitk.GetArrayFromImage(raw)
        drop_area = sitk.GetArrayFromImage(sitk.ReadImage(totalsegname)) > 0
        image[drop_area] = np.random.random((drop_area.sum(),)) * std + mean
        
        dropped_image = sitk.GetImageFromArray(image)
        dropped_image.CopyInformation(raw)
        sitk.WriteImage(dropped_image, join(savepath, f"OrganDrop_{imid}_0000.nii.gz"))
        shutil.copyfile(maskname, join(savepath.replace("imagesTr", "labelsTr"), f"OrganDrop_{imid}.nii.gz"))
        
        
if __name__ == "__main__":
    main()