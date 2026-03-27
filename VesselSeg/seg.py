import multiprocessing as mp
import threading
import os
import json
import torch
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from functools import partial, reduce
from helpers.utils import LabelParser, OrganTypeBase
from scipy.ndimage import binary_dilation, label
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data, get_output_folder
from nnunetv2.ensembling.ensemble import ensemble_folders
from VesselSeg.train import name_id_mapping, desc_fold_mapping

mp.set_start_method("spawn", force=True)


def deprecated(func):
    def fn(*args, **kwargs):
        print(f"I am doing nothing because you used a deprecated function: {func.__name__}")
        return 1
    return fn


def _maybe_mkdir(p, destory_on_exist=False):
    if destory_on_exist and os.path.exists(p):
        try:
            shutil.rmtree(p)
        except Exception as e:
            print(f"failed to remove existing folder {p}")
            raise e
    os.makedirs(p, exist_ok=1)
    return p


def _is_empty(p):
    return os.path.isdir(p) and (len(os.listdir(p)) == 0)


class VesselSegInstance:
    def __init__(self, gpus=[0, 0, 1, 2, 3],
                 dst_folder="/data/cache/LNM/ensemble_pred",
                 src_folder="/data/cache/LNM/image",
                 totalseg_folder="/data/cache/LNM/totalseg",
                 scheme="test",
                 n_files_per_iteration=5,
                 p_foreground=.9999,
                 start=0,
                 end=None,
                 is_pretrained=True,
                 totalseg_version='v2',
                 version=0,
                 use_mp=False):
        self.version = f"version_{version}"
        self.data = []
        self.data_mapping = {}
        self.reverse_data_mapping = {}
        self.folds = [desc_fold_mapping[f'ft{i}'] for i in range(5)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in gpus])
        _maybe_mkdir(dst_folder, destory_on_exist=0)
        if scheme == "test":
            flag = 0
            if os.path.exists(os.path.join(dst_folder, "nnunet_id2dataset_id.json")):
                with open(os.path.join(dst_folder, "nnunet_id2dataset_id.json")) as f, open(os.path.join(dst_folder, "dataset_id2nnunet_id.json")) as g:
                    self.data_mapping = json.load(g)
                    self.reverse_data_mapping = json.load(f)
            else:
                flag = 1
                
            if os.path.exists(os.path.join(dst_folder, "finals")):
                done_cases = os.listdir(os.path.join(dst_folder, "finals"))
            else:
                done_cases = []
                
            if os.path.isfile(os.path.join(dst_folder, src_folder)):
                with open(os.path.join(dst_folder, src_folder)) as f:
                    lines = f.readlines()
            else:
                lines = [os.path.join(src_folder, _) for _ in os.listdir(src_folder)]
            for iline, line in enumerate(lines):
                if not f"Dataset200LYMPH_{iline:05}.nii.gz" in done_cases:
                    self.data.append(line.strip())
                if flag == 1:
                    self.data_mapping[os.path.basename(line.strip())] = f"Dataset200Dummy_{iline:05}_0000.nii.gz"
                    self.reverse_data_mapping[f"Dataset200Dummy_{iline:05}_0000.nii.gz"] = line.strip().split('/')[-1]
                with open(os.path.join(dst_folder, "nnunet_id2dataset_id.json"), 'w') as f, \
                    open(os.path.join(dst_folder, "dataset_id2nnunet_id.json"), 'w') as g:
                    json.dump(self.reverse_data_mapping, f, indent=4)
                    json.dump(self.data_mapping, g, indent=4)
                
        self.dst = dst_folder
        self.gpus = gpus
        self.start = start
        self.is_pretrained = is_pretrained
        self.end = end if end is not None else len(self.data)
        self.scheme = scheme
        self.p_foreground = p_foreground
        self.n_files_per_iteration = n_files_per_iteration
        self.parser = LabelParser(totalseg_version)
        
        if scheme == "test":
            # create temporary folders for intermediate results
            self.inputs_folder = _maybe_mkdir(os.path.join(self.dst, "inputs_" + self.version), destory_on_exist=1)
            self.preds_folder1 = _maybe_mkdir(os.path.join(self.dst, "preds1_" + self.version), destory_on_exist=1)
            self.preds_folder2 = _maybe_mkdir(os.path.join(self.dst, "preds2_" + self.version), destory_on_exist=1)
            self.preds_buffer = _maybe_mkdir(os.path.join(self.dst, "preds_buffer"), destory_on_exist=1)
            [_maybe_mkdir(os.path.join(self.dst, "preds_buffer", f"fold_{i}"), destory_on_exist=1) for i in range(5)]
            self.ensemble_folder = _maybe_mkdir(os.path.join(self.dst, "ensembles_" + self.version), destory_on_exist=1)
            self.ensemble_buffer = _maybe_mkdir(os.path.join(self.dst, "ensembles_buffer"), destory_on_exist=1)
            self.outputs_folder = _maybe_mkdir(os.path.join(self.dst, "finals"),)
            self.totalseg_folder = _maybe_mkdir(os.path.join(self.dst, "totalsegs"))
            if totalseg_folder is not None:
                for file in os.listdir(totalseg_folder):
                    shutil.copyfile(os.path.join(totalseg_folder, file), os.path.join(self.totalseg_folder, self.data_mapping[file]))
            
            # use threading-based synchronization within a single process to avoid
            # pickling issues with multiprocessing Semaphores under spawn
            self.predict_lock = threading.Semaphore(2)
            self.ensemble_lock = threading.Semaphore(0)
            self.is_finished = threading.Semaphore(0)
            self.test_pipeline(use_mp)
        else:
            raise NotImplementedError("currently only support prediction")
        
    def cleanse(self):
        print("cleansing temp folders ...")
        _maybe_mkdir(self.inputs_folder, destory_on_exist=1)
        _maybe_mkdir(self.preds_folder1, destory_on_exist=1)
        _maybe_mkdir(self.preds_folder2, destory_on_exist=1)
        _maybe_mkdir(self.ensemble_folder, destory_on_exist=1)
    
    def test_pipeline(self, use_mp=False):
        try:
            if use_mp:
                # run predict and ensemble in separate threads within this process
                self.predict_thread = threading.Thread(target=self._predict, daemon=True)
                self.ensemble_thread = threading.Thread(target=self._ensemble, daemon=True)
                self.predict_thread.start()
                self.ensemble_thread.start()
                self.predict_thread.join()
                self.ensemble_thread.join()
            else:
                self.predict_and_ensemble()
        except (KeyboardInterrupt, SystemExit, RuntimeError, NotImplementedError) as e:
            print(e)
        finally:
            # self.cleanse()
            pass
    
    @deprecated
    def _predict_singlefile(self):
        for file_start_index in range(self.start, self.end, self.n_files_per_iteration):
            self.predict_lock.acquire()
            preds_folder = self.preds_folder1
            assert _is_empty(preds_folder), f"{preds_folder} is not empty"
            # load data
            for file in self.data[file_start_index: file_start_index + self.n_files_per_iteration]:
                shutil.copyfile(file, os.path.join(self.inputs_folder, self.data_mapping[os.path.basename(file)]))
            # predict
            predict_ = partial(predict_from_raw_data, 
                               list_of_lists_or_source_folder=self.inputs_folder,
                               save_probabilities=1,
                               checkpoint_name="checkpoint_best.pth")
            pool = []
            for fold, device, ofolder in zip(self.folds, 
                                            (self.gpus[0], self.gpus[0], self.gpus[0], self.gpus[1], self.gpus[1]), 
                                            tuple(os.path.join(preds_folder, f"fold_{i}") for i in range(5))):
                _maybe_mkdir(os.path.join(preds_folder, f"fold_{fold}"))
                pool.append(mp.Process(target=predict_, kwargs=dict(
                    model_training_output_dir=get_output_folder(name_id_mapping['normalloss'], 'nnUNetTrainer', 'nnUNetPlans', '3d_fullres', desc='all'),
                    output_folder=ofolder,
                    use_folds=[fold],
                    device=torch.device('cuda', device))))
                pool[-1].start()
            
            for p in pool:
                p.join()
            
            _maybe_mkdir(self.inputs_folder, destory_on_exist=1)
            self.ensemble_lock.release()
            # yield preds_folder
            
        self.is_finished.release()

    @deprecated  
    def _ensemble_singlefile(self):
        for _ in range(self.start, self.end, self.n_files_per_iteration):
            self.ensemble_lock.acquire()
            preds_folder = self.preds_folder1
            assert not _is_empty(self.preds_folder1), f"{preds_folder} is empty"
                
            ensemble_folders(tuple(os.path.join(preds_folder, f"fold_{i}") for i in self.folds), self.ensemble_folder, save_merged_probabilities=1)
            _maybe_mkdir(preds_folder, destory_on_exist=1)
            self.predict_lock.release()
            
            # postprocessing
            ensembled = sorted([os.path.join(self.ensemble_folder, _) for _ in os.listdir(self.ensemble_folder) if _.endswith(".npz")],
                               key=lambda x: os.path.basename(x))
            print("postprocessing cases:" + "\n".join(ensembled))
            for fn in ensembled:
                f = np.load(fn)["probabilities"]
                nnunet_id = os.path.basename(fn).replace(".npz", ".nii.gz")
                tn = os.path.join(self.totalseg_folder, nnunet_id)
                
                ti = sitk.ReadImage(tn)
                t = sitk.GetArrayFromImage(ti)
                roi = binary_dilation(reduce(lambda x, y: x | y, [t == 18, t == 20, t == 52, t == 63]), iterations=2) 
                
                f[0, f[0] <= self.p_foreground] = 0
                preliminary_result = f.argmax(0)
                result_im = sitk.GetImageFromArray(preliminary_result.astype(np.uint8))
                result_im.CopyInformation(ti)
                sitk.WriteImage(result_im, fileName=os.path.join(self.outputs_folder, os.path.basename(tn).replace(".nii.gz", f"_prob_{self.p_foreground}_not_postprocessed.nii.gz")))
                
                del f
                result = np.zeros_like(preliminary_result)
                labeled_result, n = label(preliminary_result)
                for i in range(1, n):
                    if ((labeled_result == i) & roi).sum() > 100:
                        result[labeled_result == i] = preliminary_result[labeled_result == i]
                
                result_im = sitk.GetImageFromArray(result.astype(np.uint8))
                result_im.CopyInformation(ti)
                sitk.WriteImage(result_im, fileName=os.path.join(self.outputs_folder, os.path.basename(tn)))
                print(f"wrote postprocessed result to {os.path.join(self.outputs_folder, os.path.basename(tn))}")
            
            _maybe_mkdir(self.ensemble_folder, destory_on_exist=1)

    @deprecated 
    def _predict(self):
        for file_start_index in range(self.start, self.end, self.n_files_per_iteration):
            self.predict_lock.acquire()
            if _is_empty(self.preds_folder1): preds_folder = self.preds_folder1
            elif _is_empty(self.preds_folder2): preds_folder = self.preds_folder2
            else: raise RuntimeError("both preds folder are not empty")
            # load data
            for file in self.data[file_start_index: file_start_index + self.n_files_per_iteration]:
                shutil.copyfile(file, os.path.join(self.inputs_folder, self.data_mapping[os.path.basename(file)]))
            # predict
            predict_ = partial(predict_from_raw_data, 
                               list_of_lists_or_source_folder=self.inputs_folder,
                               save_probabilities=1,
                               checkpoint_name="checkpoint_best.pth")
            pool = []
            for i, fold, device, ofolder in zip(range(5),
                                                self.folds, 
                                                (self.gpus[0], self.gpus[1], self.gpus[2], self.gpus[3], self.gpus[4]), 
                                                tuple(os.path.join(preds_folder, f"fold_{i}") for i in range(5))):
                _maybe_mkdir(os.path.join(preds_folder, f"fold_{fold}"))
                pool.append(mp.Process(target=predict_, kwargs=dict(
                    model_training_output_dir=get_output_folder(
                        name_id_mapping['normalloss'], 
                        'nnUNetTrainer', 
                        'nnUNetPlans', 
                        '3d_fullres', 
                        desc=f"all_ft{i}_{'finetune' if self.is_pretrained else 'scratch'}"
                    ),
                    output_folder=ofolder,
                    use_folds=[fold],
                    device=torch.device('cuda', device))))
                pool[-1].start()
            
            for p in pool:
                p.join()
            
            _maybe_mkdir(self.inputs_folder, destory_on_exist=1)
            self.ensemble_lock.release()
            # yield preds_folder
            
        self.is_finished.release()
    
    @deprecated
    def _ensemble(self):
        # ensemble
        for _ in range(self.start, self.end, self.n_files_per_iteration):
            self.ensemble_lock.acquire()
            if _is_empty(self.preds_folder1): 
                self.predict_lock.release()
                if not _is_empty(self.preds_folder2):
                    preds_folder = self.preds_folder2
                else: 
                    self.is_finished.acquire()
                    break
            else:
                preds_folder = self.preds_folder1
                if _is_empty(self.preds_folder2):
                    self.predict_lock.release()
                
            ensemble_folders(tuple(os.path.join(preds_folder, f"fold_{i}") for i in range(5)), self.ensemble_folder, save_merged_probabilities=1)
            for file in Path(preds_folder).rglob('*.nii.gz'):
                buffer_dir = Path(self.preds_buffer) / file.parent.name
                buffer_dir.mkdir(parents=True, exist_ok=True)
                buffer_path = buffer_dir / Path(self.reverse_data_mapping[file.name]).name
                shutil.copyfile(file, buffer_path)
            _maybe_mkdir(preds_folder, destory_on_exist=1)
            self.predict_lock.release()
            
            # postprocessing
            ensembled = sorted([os.path.join(self.ensemble_folder, _) for _ in os.listdir(self.ensemble_folder) if _.endswith(".npz")],
                               key=lambda x: os.path.basename(x))
            print("postprocessing cases:" + "\n".join(ensembled))
            for fn in ensembled:
                f = np.load(fn)["probabilities"]
                nnunet_id = os.path.basename(fn).replace(".npz", ".nii.gz")
                tn = os.path.join(self.totalseg_folder, nnunet_id)
                
                ti = sitk.ReadImage(tn)
                t = sitk.GetArrayFromImage(ti)
                # roi = binary_dilation(reduce(lambda x, y: x | y, [t == 18, t == 20, t == 52, t == 63]), iterations=2) 
                roi = self.parser.totalseg2mask(
                    t, 
                    [
                        OrganTypeBase(name='colon', label=1), 
                        OrganTypeBase(name='small_bowel', label=1),
                        OrganTypeBase(name='aorta', label=1),
                        OrganTypeBase(name='inferior_vena_cava', label=1)
                    ]
                )
                
                f[0, f[0] <= self.p_foreground] = 0
                preliminary_result = f.argmax(0)
                result_im = sitk.GetImageFromArray(preliminary_result.astype(np.uint8))
                result_im.CopyInformation(ti)
                sitk.WriteImage(result_im, fileName=os.path.join(self.outputs_folder, os.path.basename(tn).replace(".nii.gz", f"_prob_{self.p_foreground}_not_postprocessed.nii.gz")))
                
                del f
                result = np.zeros_like(preliminary_result)
                labeled_result, n = label(preliminary_result)
                for i in range(1, n):
                    if ((labeled_result == i) & roi).sum() > 100:
                        result[labeled_result == i] = preliminary_result[labeled_result == i]
                
                result_im = sitk.GetImageFromArray(result.astype(np.uint8))
                result_im.CopyInformation(ti)
                sitk.WriteImage(result_im, fileName=os.path.join(self.outputs_folder, os.path.basename(tn)))
                print(f"wrote postprocessed result to {os.path.join(self.outputs_folder, os.path.basename(tn))}")
            
            for file in Path(preds_folder).rglob('*.nii.gz'):
                buffer_path = Path(self.ensemble_buffer) / Path(self.reverse_data_mapping[file.name]).name
                buffer_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, buffer_path)
            _maybe_mkdir(self.ensemble_folder, destory_on_exist=1)
                  
    def predict_and_ensemble(self):
        for file_start_index in range(self.start, self.end, self.n_files_per_iteration):
            # clean folders
            self.cleanse()
            preds_folder = self.preds_folder1
            # load data
            for file in self.data[file_start_index: file_start_index + self.n_files_per_iteration]:
                shutil.copyfile(file, os.path.join(self.inputs_folder, self.data_mapping[os.path.basename(file)]))
            # predict
            pool = []
            predict_ = partial(predict_from_raw_data, 
                               list_of_lists_or_source_folder=self.inputs_folder,
                               save_probabilities=1,
                               checkpoint_name="checkpoint_final.pth")
            for i, fold, device, ofolder in zip(range(5),
                                                self.folds, 
                                                (self.gpus[0], self.gpus[1], self.gpus[2], self.gpus[3], self.gpus[4]), 
                                                tuple(os.path.join(preds_folder, f"fold_{i}") for i in range(5))):
                _maybe_mkdir(os.path.join(preds_folder, f"fold_{i}"))
                model_training_output_dir = get_output_folder(
                    name_id_mapping['normalloss'], 
                    'nnUNetTrainer', 
                    'nnUNetPlans', 
                    '3d_fullres', 
                    desc=f"{'all_' if self.is_pretrained else ''}ft{i}_{'finetune' if self.is_pretrained else 'scratch'}"
                )
                pool.append(mp.Process(target=predict_, kwargs=dict(
                    model_training_output_dir=model_training_output_dir,
                    output_folder=ofolder,
                    use_folds=[fold],
                    device=torch.device('cuda', device),
                    num_processes_segmentation_export=1
                )))
                pool[-1].start()

            for p in pool:
                p.join()
            
            ensemble_folders(
                tuple(os.path.join(preds_folder, f"fold_{i}") for i in range(5)),
                self.ensemble_folder,
                save_merged_probabilities=1
            )
            for file in Path(preds_folder).rglob('*.nii.gz'):
                shutil.copyfile(
                    file, 
                    os.path.join(
                        self.preds_buffer,
                        file.parent.name,
                        self.reverse_data_mapping[file.name.replace('.nii.gz', '_0000.nii.gz')]))
            
            # postprocessing
            ensembled = sorted(
                [os.path.join(self.ensemble_folder, _) for _ in os.listdir(self.ensemble_folder) if _.endswith(".npz")], key=lambda x: os.path.basename(x)
            )
            print("postprocessing cases:" + "\n".join(ensembled))
            for fn in ensembled:
                f = np.load(fn)["probabilities"]
                nnunet_id = os.path.basename(fn).replace(".npz", "_0000.nii.gz")
                tn = os.path.join(self.totalseg_folder, nnunet_id)
                
                ti = sitk.ReadImage(tn)
                t = sitk.GetArrayFromImage(ti)
                # roi = binary_dilation(reduce(lambda x, y: x | y, [t == 18, t == 20, t == 52, t == 63]), iterations=2) 
                roi = self.parser.totalseg2mask(
                    t, 
                    [
                        OrganTypeBase(name='colon', label=1), 
                        OrganTypeBase(name='smallbowel', label=1),
                        OrganTypeBase(name='aorta', label=1),
                        OrganTypeBase(name='inferiorvenacava', label=1)
                    ]
                )
                for p_foreground in [0, 1-1e-3]:
                    f[0, f[0] <= p_foreground] = 0
                    preliminary_result = f.argmax(0)
                    
                    result_im = sitk.GetImageFromArray(preliminary_result.astype(np.uint8))
                    result_im.CopyInformation(ti)
                    sitk.WriteImage(
                        result_im, 
                        fileName=os.path.join(
                            self.outputs_folder,
                            self.reverse_data_mapping[nnunet_id].replace(".nii.gz", f"_prob_{p_foreground}_boosted.nii.gz")
                        ))
                
                    result = np.zeros_like(preliminary_result)
                    labeled_result, n = label(preliminary_result)
                    for i in range(1, n):
                        if (binary_dilation(labeled_result == i, iterations=3) & roi).sum() > 100:
                            result[labeled_result == i] = preliminary_result[labeled_result == i]
                    
                    result_im = sitk.GetImageFromArray(result.astype(np.uint8))
                    result_im.CopyInformation(ti)
                    sitk.WriteImage(
                        result_im, 
                        fileName=os.path.join(
                            self.outputs_folder, 
                            self.reverse_data_mapping[nnunet_id].replace(".nii.gz", f"_prob_{p_foreground}_post.nii.gz")
                        ))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=[0] * 5, type=int, nargs='+')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--nproc", default=1, type=int)
    parser.add_argument("--nfile_per_iter", default=2, type=int)
    parser.add_argument("--no_pretrain", action='store_true')
    parser.add_argument("--p_foreground", default=0.999, type=float)
    parser.add_argument("--save_dir", default="/data/cache/LNM/", type=str)
    parser.add_argument("--image_dir", default="/data/cache/LNM/image", type=str)
    parser.add_argument("--totalseg_dir", default="/data/cache/LNM/totalseg", type=str)
    args, unk = parser.parse_known_args()

    if len(args.gpu) > 5: args.gpu = args.gpu[:5]
    elif len(args.gpu) < 5: args.gpu = args.gpu + [args.gpu[-1]] * (5 - len(args.gpu))

    VesselSegInstance(
        gpus=args.gpu,
        n_files_per_iteration=args.nfile_per_iter,
        src_folder=args.image_dir,
        dst_folder=args.save_dir,
        totalseg_folder=args.totalseg_dir,
        is_pretrained=not args.no_pretrain,
        p_foreground=args.p_foreground,
        start=args.start
    )