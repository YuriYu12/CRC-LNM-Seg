import torch
import os


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


def empty_cache(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        from torch import mps
        mps.empty_cache()
    else:
        pass


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def parse_new_training_scheme(output_folder_base, fold, desc=None, findex=None):
    # the original nnunetv2-named folder name for saving training chk is
    # $output_folder_base/foldY/...
    # if there is no desc, we rename it as
    # $output_folder_base/foldY_<time of its creation>_<seq index>/...
    # if there is desc, we rename it as
    # $output_folder_base/foldY_$desc
    
    all_subdirs = os.listdir(output_folder_base)
    if desc is not None:
        desc_subdirs = [subdir for subdir in all_subdirs if f"fold{fold}" in subdir and desc in subdir]
        assert len(desc_subdirs) == 1, f"found all subdirs under desc {desc}: {desc_subdirs}, expected only 1 subdir"
        assert findex is None, "cannot assign desc and fname at the same time"
    if findex is not None:
        pass