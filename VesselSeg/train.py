import os
import json
import math
import copy
import torch
import random
import shutil
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from medpy.metric import binary
from collections import defaultdict
from scipy.ndimage import binary_dilation, label as ndimage_label
from nnunetv2.run.run_training import run_training
from helpers.utils import LabelParser, OrganTypeBase
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data, get_output_folder
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess as nnunet_preprocess
from skimage.morphology import skeletonize
import numpy as np


def cldice(v_p, v_l):
    def cl_score(v, s):
        return np.sum(v*s)/np.sum(s)
    if len(v_p.shape)==2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    return 2*tprec*tsens/(tprec+tsens)


class Mapper:
    def __init__(self):
        self.mapping = {
            "100syn": 0,
            "200syn": 1,
            "300syn": 2,
            "400syn": 3,
            "500syn": 4,
            "600syn": 5,
            "all": 6,
            "real": 7,
            "ft0": 8,
            "ft1": 9,
            "ft2": 10,
            "ft3": 11,
            "ft4": 12,
        }
        
    def __len__(self):
        return len(self.mapping)
        
    def __getitem__(self, name):
        x = self.mapping.get(name.split('_')[-1])
        return x
    
desc_fold_mapping = Mapper()  
name_id_mapping = {
    "normalloss": 105,
}

def multiclass_metrics_wrapper(fn, n, include_bg_metrics=False, hd95_infty=1000):
    def _impl(x, y, **kw):
        ret = {}
        for i in range(0 if include_bg_metrics else 1, n):
            if fn == 'cldice':
                fn_ = cldice
            else:   
                fn_ = getattr(binary, fn)
            if (x == i).sum() == 0: ret[i] = np.nan if fn not in ('hd95', 'hd') else hd95_infty
            elif (y == i).sum() == 0: ret[i] = 0 if fn not in ('hd95', 'hd') else hd95_infty
            else: ret[i] = fn_(x == i, y == i, **kw)
        return ret
    return _impl


def maybe_mkdir(path, destroy_on_exist=False):
    if os.path.exists(path) and destroy_on_exist: shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return Path(path)


def create_link(src, dst, use_hard=False):
    if os.path.exists(dst):
       os.remove(dst)
    os.symlink(src, dst) if not use_hard else shutil.copyfile(src, dst)
    return dst 


def window_trunc(arr, a):
    a_min, a_max = a
    arr[arr < a_min] = a_min
    arr[arr > a_max] = a_max
    return arr


def make_train_test(name='normalloss',
                    dataset_id=105,
                    base="/data/dataset",):
    task_name = f"Dataset{dataset_id:03d}_{name}"
    task_ds_path = maybe_mkdir(os.path.join(nnUNet_raw, task_name))
    task_ds_tr_image_path = maybe_mkdir(os.path.join(nnUNet_raw, task_name, "imagesTr"), destroy_on_exist=True)
    task_ds_tr_label_path = maybe_mkdir(os.path.join(nnUNet_raw, task_name, "labelsTr"), destroy_on_exist=True)
    task_ds_ts_image_path = maybe_mkdir(os.path.join(nnUNet_raw, task_name, "imagesTs"), destroy_on_exist=True)
    task_ds_ts_label_path = maybe_mkdir(os.path.join(nnUNet_raw, task_name, "labelsTs"), destroy_on_exist=True)
    
    task_train_set_organized = zip(sorted(list((Path(base) / 'imagesTr').glob("*.nii.gz"))), sorted(list((Path(base) / 'labelsTr').glob("*.nii.gz"))))
    task_test_set_organized = zip(sorted(list((Path(base) / 'imagesTs').glob("*.nii.gz"))), sorted(list((Path(base) / 'labelsTs').glob("*.nii.gz"))))
        
    nnunet2real_mapping = defaultdict(dict)
    splits = [{"train": [], "val": []} for _ in range(len(desc_fold_mapping))]
    dataset = {"channel_names": {0: "CT"},
               'labels': {"background": 0,
                          "sup_mesenteric_1": 1,
                          "inf_mesenteric_1": 2,
                          "sup_mesenteric_2": 3,
                          "inf_mesenteric_2": 4,
                          "sup_mesenteric_3": 5,
                          "inf_mesenteric_3": 6,},
               'file_ending': ".nii.gz"}
    
    i = 0
    for image, label in tqdm(task_train_set_organized, total=len(list((Path(base) / 'imagesTr').glob("*.nii.gz"))), desc="syn set organize progress"):
        i += 1
        nnunet_id = f"{task_name}_{i:05d}"
        nnunet2real_mapping[nnunet_id]['image'] = str(image)
        nnunet2real_mapping[nnunet_id]['label'] = str(label)
        splits[desc_fold_mapping['all']]["train"].append(nnunet_id)
        create_link(image, task_ds_tr_image_path / f"{nnunet_id}_0000.nii.gz")
        create_link(label, task_ds_tr_label_path / f"{nnunet_id}.nii.gz")
        if i >= 700: break
    
    j = 0
    for image, label in tqdm(task_test_set_organized, total=len(list((Path(base) / 'imagesTs').glob("*.nii.gz"))), desc="real disease set organize progress"):
        i += 1
        j += 1
        nnunet_id = f"{task_name}_{i:05d}"
        nnunet2real_mapping[nnunet_id]['image'] = str(image)
        nnunet2real_mapping[nnunet_id]['label'] = str(label)
        [splits[_]["val"].append(nnunet_id) for _ in range(len(splits) - 5)]
        splits[desc_fold_mapping['real']]['train'].append(nnunet_id)
        create_link(image, task_ds_tr_image_path / f"{nnunet_id}_0000.nii.gz")
        create_link(image, task_ds_ts_image_path / f"{nnunet_id}_0000.nii.gz")
        create_link(label, task_ds_tr_label_path / f"{nnunet_id}.nii.gz")
        create_link(label, task_ds_ts_label_path / f"{nnunet_id}.nii.gz")

    dataset["numTraining"] = i
    for n in [100, 200, 300, 400, 500, 600]:
        splits[desc_fold_mapping[f"{n}syn"]]['train'] = splits[desc_fold_mapping['all']]['train'][:n]
    randomized_real = copy.deepcopy(splits[desc_fold_mapping['real']]['train'])
    random.shuffle(randomized_real)
    for cv in range(5):
        splits[desc_fold_mapping[f'ft{cv}']]['train'] = [case for case in splits[desc_fold_mapping['real']]['train'] if case not in randomized_real[cv * 2: (cv + 1) * 2]]
        splits[desc_fold_mapping[f'ft{cv}']]['val'] = randomized_real[cv * 2: (cv + 1) * 2]
        
    with open(task_ds_path / "dataset.json", 'w') as f,\
        open(task_ds_path / "nnunet2real.json", 'w') as g:
        json.dump(dataset, f, indent=4)
        json.dump(dict(**nnunet2real_mapping), g, indent=4)
    
    maybe_mkdir(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id)))
    with open(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id), "splits_final.json"), 'w') as f:
        json.dump(splits, f, indent=4)
    
    return splits


def preprocess(dataset_id, name):
    print('Fingerprints extracting...')
    extract_fingerprints([dataset_id], check_dataset_integrity=True, )
    print('Experiment planning...')
    plan_experiments([dataset_id], )
    nnunet_preprocess([dataset_id],  num_processes=[8], configurations=['3d_fullres'])
    
    with open(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id), "nnUNetPlans.json"), 'r') as f:
        raw = json.load(f)
    if name == 'normalloss': loss_name = 'ce_and_dice'
    elif name == 'proposedloss1': loss_name = 'weighted_ce_and_dice'
    elif name == 'proposedloss2': loss_name = 'delta_ce_and_dice'
    elif name == 'proposedlossall': loss_name = 'delta_weighted_ce_and_dice'
    raw['configurations']['3d_fullres']['loss_name'] = loss_name
    with open(os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id), "nnUNetPlans.json"), 'w') as f:
        json.dump(raw, f, indent=4)
    
    
def train(dataset_id, resume=False, desc="", fold=0, epochs=1000, start_lr=1e-2, no_pretrain=False):
    if not ('ft' in desc or 'finetune' in desc):
        print("pretraining")
        run_training(str(dataset_id), configuration='3d_fullres', fold=fold, max_epoch=epochs, start_lr=start_lr, desc=desc, continue_training=resume)
    elif no_pretrain:
        print("finetuning from scratch")
        run_training(str(dataset_id), configuration='3d_fullres', fold=fold, max_epoch=epochs, start_lr=start_lr*0.1, desc=f'{desc}_scratch', continue_training=resume)
    else:
        print("finetuning")
        run_training(str(dataset_id), configuration='3d_fullres', fold=fold, max_epoch=epochs, start_lr=start_lr*0.1, desc=f'{desc}_finetune',
                     pretrained_weights=os.path.join(get_output_folder(dataset_id, desc=desc.split('_ft')[0], fold=desc_fold_mapping[desc.split('_ft')[0]]), "checkpoint_final.pth"))
    
    
def test(dataset_id, imagesTf, labelsTf, predsTf, desc="", n_classes=7, only_metric_cal=False, fold=0, all_metrics=['dc', 'cldice'], use_postprocess=False, fg_prob=0.0001):
    if not only_metric_cal:
        predict_from_raw_data(str(imagesTf), str(predsTf), 
                              get_output_folder(dataset_id, desc=desc), 
                              use_folds=(fold,), 
                              checkpoint_name="checkpoint_final.pth",
                              save_probabilities=True)
    if True:
        npzs = sorted(list(Path(predsTf).glob("*.npz")))
        preds = sorted(list(Path(predsTf).glob("*.nii.gz")))
        totalsegs = [Path(str(labelsTf).replace('labelsTf', 'totalsegTs')).parent / _.name.replace('.npz', '.nii.gz') for _ in npzs]

        parser = LabelParser('v2')
        for pred, npz, totalseg in tqdm(zip(preds, npzs, totalsegs), total=len(npzs), desc="boost progress"):
            probs = np.load(npz)['probabilities']
            boost = probs.copy()
            boost[0, boost[0] < 1 - fg_prob] = 0
            boost_label = boost.argmax(0).astype(np.uint8)
            
            # if totalseg is not None:
            #     totalseg = sitk.GetArrayFromImage(sitk.ReadImage(totalseg))
            #     abd = parser.totalseg2mask(totalseg, [
            #         OrganTypeBase(name='colon', label=1), 
            #         OrganTypeBase(name='smallbowel', label=1),
            #         OrganTypeBase(name='aorta', label=1), 
            #         OrganTypeBase(name='inferiorvenacava', label=1),
            #     ])
            #     result = np.zeros_like(boost_label)
            #     labeled_result, n = ndimage_label(boost_label)
            #     for i in range(1, n):
            #         if (binary_dilation(labeled_result == i, iterations=3) & abd).sum() > 100:
            #             result[labeled_result == i] = boost_label[labeled_result == i]
            
            im = sitk.GetImageFromArray(boost_label)
            im.CopyInformation(sitk.ReadImage(pred))
            sitk.WriteImage(im, str(pred).replace('.nii.gz', '_post.nii.gz'))

    for npz in npzs:
        os.remove(npz)
    
    metrics = {}
    load = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    for image in tqdm(Path(imagesTf).glob("*.nii.gz"), total=len(list(Path(imagesTf).glob("*.nii.gz")))):
        label = load(str(image).replace(str(imagesTf), str(labelsTf)).replace("_0000.nii.gz", ".nii.gz"))
        pred = load(str(image).replace(str(imagesTf), str(predsTf)).replace("_0000.nii.gz", ".nii.gz"))
        metrics[image.name] = {metric: multiclass_metrics_wrapper(metric, n_classes)(pred, label) for metric in all_metrics}
            
    classwise_metrics = {metric: {i: np.mean([x for _ in metrics.keys() if not np.isnan(x:=metrics[_][metric].get(i, np.nan))]) for i in range(0, 7)} for metric in all_metrics}
    for image in tqdm(Path(imagesTf).glob("*.nii.gz"), total=len(list(Path(imagesTf).glob("*.nii.gz")))):
        label = load(str(image).replace(str(imagesTf), str(labelsTf)).replace("_0000.nii.gz", ".nii.gz"))
        pred = load(str(image).replace(str(imagesTf), str(predsTf)).replace("_0000.nii.gz", "_post.nii.gz"))
        metrics[image.name] |= {f"{metric}_post": multiclass_metrics_wrapper(metric, n_classes)(pred, label) for metric in all_metrics}
            
    classwise_metrics = {metric: {i: np.mean([x for _ in metrics.keys() if not np.isnan(x:=metrics[_][metric].get(i, np.nan))]) for i in range(0, 7)} for metric in all_metrics}|\
                        {f"{metric}_post": {i: np.mean([x for _ in metrics.keys() if not np.isnan(x:=metrics[_][f'{metric}_post'].get(i, np.nan))]) for i in range(0, 7)} for metric in all_metrics}
    with open(Path(predsTf) / f"metrics_{desc}.json", 'w') as f:
        json.dump({"instancewise": metrics, "classwise": classwise_metrics},
                    f, ensure_ascii=False, indent=4)
    
    print(json.dumps(classwise_metrics, indent=4))
    

def pipeline(args):
    name = args.name
    is_preprocess = args.preprocess
    is_train = args.train
    is_test = args.test
    desc = args.desc
    use_sbatch = not args.no_sbatch
    no_pretrain = args.no_pretrain
    
    if name.isnumeric(): name = [k for k in name_id_mapping.keys() if name_id_mapping[k] == int(name)][0]
    fold = desc_fold_mapping[desc]
    dataset_id = name_id_mapping[name]
    if is_preprocess:
        print("preprocessing")
        if not args.made_train_test: make_train_test(name=name, dataset_id=dataset_id, base=args.preprocess_dir)
        preprocess(dataset_id=dataset_id, name=name)
    if is_train:
        print("training")
        if not use_sbatch: train(dataset_id=dataset_id, desc=desc, fold=fold, no_pretrain=no_pretrain)
        else:
            print("sbatch "+\
                f"-D {os.path.dirname(__file__)} "+\
                f"-J nnunet_{name}_seg_{desc} "+\
                f"-o {os.path.join(maybe_mkdir(os.path.dirname(get_output_folder(dataset_id, desc=desc))), 'out.txt')} "+\
                f"-p smart_health_02 "+\
                "-N 1 "+\
                "-n 1 "+\
                "--cpus-per-task=4 "+\
                "--gpus=1 "+\
                f"--mem={96 if 'ft' not in desc else 128}G "+\
                f"--wrap \"python train.py --name {name} --train --desc {desc} --fold {fold} --no_sbatch\""
                )
    if is_test:
        print(f'testing, metric only={args.metric_only}')
        input_image_dir = maybe_mkdir(os.path.join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_id), f"imagesTf", f"fold_{fold}"))
        input_label_dir = maybe_mkdir(os.path.join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_id), f"labelsTf", f"fold_{fold}"))
        fold_file = os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_id), "splits_final.json")
        with open(fold_file, 'r') as f:
            splits = json.load(f)
        for case in splits[fold]['val']:
            src_image = os.path.join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_id), "imagesTs", f"{case}_0000.nii.gz")
            dst_image = os.path.join(input_image_dir, f"{case}_0000.nii.gz")
            create_link(src_image, dst_image)
            src_label = os.path.join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_id), "labelsTs", f"{case}.nii.gz")
            dst_label = os.path.join(input_label_dir, f"{case}.nii.gz")
            create_link(src_label, dst_label)
        preds_label_dir = maybe_mkdir(os.path.join(nnUNet_raw, maybe_convert_to_dataset_name(dataset_id), f"predsTf", f"fold_{fold}"), destroy_on_exist=not args.metric_only)
        desc = f"{desc}_finetune" if not args.no_pretrain else f"{desc}_scratch"
        test(dataset_id=dataset_id, desc=desc, only_metric_cal=args.metric_only, fold=fold,
             imagesTf=input_image_dir, labelsTf=input_label_dir, predsTf=preds_label_dir,
             use_postprocess=args.postprocess, fg_prob=args.prob_fg)
        
        
def standalone_test(dataset_id, 
                    fold,
                    input_image_dir,
                    output_label_dir=None,
                    input_totalseg_dir=None,
                    desc="", fg_prob=1e-3, only_boost=True):
    output_label_dir = output_label_dir or input_image_dir
    input_totalseg_dir = input_totalseg_dir or input_image_dir
    if not only_boost:
        predict_from_raw_data(str(input_image_dir), str(output_label_dir), 
                                get_output_folder(dataset_id, desc=desc), 
                                use_folds=(fold,), 
                                checkpoint_name="checkpoint_final.pth",
                                save_probabilities=True)
        
    npzs = sorted(list(Path(output_label_dir).glob("*.npz")))
    preds = sorted(list(Path(output_label_dir).glob("*.nii.gz")))
    totalsegs = sorted(list(Path(input_totalseg_dir).glob("*.nii.gz")))

    parser = LabelParser('v2')
    for pred, npz, totalseg in tqdm(zip(preds, npzs, totalsegs), total=len(npzs), desc="boost progress"):
        probs = np.load(npz)['probabilities']
        boost = probs.copy()
        boost[0, boost[0] < 1 - fg_prob] = 0
        boost_label = boost.argmax(0).astype(np.uint8)
        
        if totalseg is not None:
            totalseg = sitk.GetArrayFromImage(sitk.ReadImage(totalseg))
            abd = parser.totalseg2mask(totalseg, [
                OrganTypeBase(name='colon', label=1), 
                OrganTypeBase(name='smallbowel', label=1),
                OrganTypeBase(name='aorta', label=1), 
                OrganTypeBase(name='inferiorvenacava', label=1),
            ])
            result = np.zeros_like(boost_label)
            labeled_result, n = ndimage_label(boost_label)
            for i in range(1, n):
                if (binary_dilation(labeled_result == i, iterations=3) & abd).sum() > 100:
                    result[labeled_result == i] = boost_label[labeled_result == i]
        
        im = sitk.GetImageFromArray(boost_label)
        im.CopyInformation(sitk.ReadImage(pred))
        sitk.WriteImage(im, str(pred).replace('.nii.gz', '_post.nii.gz'))
        

def standalone_boosting(input_label_dir, 
                        output_label_dir=None, input_totalseg_dir=None, fg_prob=0.0001, totalseg_version='v1'):
    npzs = sorted(list(Path(input_label_dir).rglob("*.npz")))
    labels = sorted(list(Path(input_label_dir).rglob("*.nii.gz")))
    totalsegs = sorted(list(Path(input_totalseg_dir).rglob("*.nii.gz"))) if input_totalseg_dir is not None else [None] * len(npzs)
    
    # boost
    parser = LabelParser(totalseg_version)
    if output_label_dir is None: output_label_dir = input_label_dir
    if not os.path.exists(output_label_dir): maybe_mkdir(output_label_dir)
    for label, npz, totalseg in tqdm(zip(labels, npzs, totalsegs), total=len(npzs), desc="boost progress"):
        probs = np.load(npz)['probabilities']
        boost = probs.copy()
        boost[0, boost[0] < 1 - fg_prob] = 0
        boost_label = boost.argmax(0).astype(np.uint8)
        
        if totalseg is not None:
            totalseg = sitk.GetArrayFromImage(sitk.ReadImage(totalseg))
            abd = parser.totalseg2mask(totalseg, [OrganTypeBase(name='colon', label=1), OrganTypeBase(name='small_bowel', label=1)])
            boost_label[~binary_dilation(abd)] = 0
        
        im = sitk.GetImageFromArray(boost_label)
        im.CopyInformation(sitk.ReadImage(label))
        sitk.WriteImage(im, os.path.join(output_label_dir, label.name))
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="normalloss")
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--preprocess_dir', type=str, default="/data/dataset")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--postprocess', action="store_true")
    parser.add_argument('--prob_fg', type=float, default=0.0001)
    parser.add_argument('--desc', type=str, default="200syn")
    parser.add_argument('--metric_only', action='store_true')
    parser.add_argument('--no_sbatch', action='store_true')
    parser.add_argument('--made_train_test', action='store_true')
    parser.add_argument('--no_pretrain', action='store_true')
    args, _ = parser.parse_known_args()
    
    pipeline(args)

    standalone_test(dataset_id=name_id_mapping["normalloss"],
                    fold=desc_fold_mapping[args.desc],
                    input_image_dir="/data/cache/LNM/image",
                    input_totalseg_dir="/data/cache/LNM/totalseg",
                    output_label_dir="/data/cache/LNM/pred",
                    fg_prob=args.prob_fg,
                    desc=f"{args.desc}_finetune" if not args.no_pretrain else f"{args.desc}_scratch")