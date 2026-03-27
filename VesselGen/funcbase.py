import os
import time
import colorama
import numpy as np
import SimpleITK as sitk

from scipy.ndimage import label
from collections.abc import Iterable
from functools import reduce, wraps


def save_nifti(array: np.ndarray, path: str, anchor_image: sitk.Image, **kwargs):
    nifti = sitk.GetImageFromArray(array)
    if anchor_image is not None:
        nifti.CopyInformation(anchor_image)
    if kwargs.__contains__("spacing"):
        nifti.SetSpacing(kwargs.get('spacing'))
    sitk.WriteImage(nifti, path)
    print(f'wrote image to {path}')
    return


def window_norm(array, typename="abdomen_vessel"):
    if typename == "abdomen_vessel":
        window_pos = 60
        window_len = 360
        
    window_lower_bound = window_pos - window_len // 2
    array = (array - window_lower_bound) / window_len
    array[array < 0] = 0
    array[array > 1] = 1
    return array


def perpendicular_dist(this_node_, start_, end_):
    end = np.array(end_)
    start = np.array(start_)
    this_node = np.array(this_node_)
    hypotenuse1 = np.linalg.norm(this_node - start)
    hypotenuse2 = np.linalg.norm(this_node - end)
    area = hypotenuse1 * hypotenuse2 * np.sqrt(max(0, 1 - ((this_node - start).T.dot(this_node - end) / (hypotenuse1 * hypotenuse2)) ** 2))
    dist = area / np.linalg.norm(end - start)
    return dist


class timeit:
    def __init__(self, funcname=None):
        self.timer = time.time()
        self.funcname = funcname
    
    def __call__(self, func):
        @wraps(func)
        def _impl(*args, **kwargs):
            st = time.time()
            retval = func(*args, **kwargs)
            printf(f"elapsed time for func {func.__name__}: {time.time() - st:.2f}s")
            return retval
        return _impl
    
    def __enter__(self):
        return self
        
    def __exit__(self, typ, val, trace):
        printf(f"elapsed time for {self.funcname}: {time.time() - self.timer:.2f}s")


class Box:
    def __init__(self, *args, **kwargs):
        self.bbox = None
        self.cropped_array = None
        self.cropped_slice = None
        
        if len(args) != 0:
            self.__call__(*args, **kwargs)
        
    def __call__(self, label_map, outline, return_slice=False, return_ndarray=False):
        # this implementation does not transfer `outline` to its correct spacing
        # should pass a spacing-rectified `outline` when calling this function
        
        if label_map is None:
            return None
        
        H, W, D = label_map.shape
        
        if outline is None:
            self.bbox = [0, H, 0, W, 0, D]
            self.cropped_array = label_map[:]
            self.cropped_slice = [slice(0, H), slice(0, W), slice(0, D)]
            
        else:
            if isinstance(outline, int):
                outline = [outline] * 6
            if len(outline) == 3:
                outline = reduce(lambda a, b: a+b, list(zip(outline, outline)))
            
            if np.sum(label_map) == 0:
                return None

            h = np.any(label_map, axis=(1, 2))
            w = np.any(label_map, axis=(0, 2))
            d = np.any(label_map, axis=(0, 1))

            hmin, hmax = np.where(h)[0][[0, -1]]
            wmin, wmax = np.where(w)[0][[0, -1]]
            dmin, dmax = np.where(d)[0][[0, -1]]
            
            retval = []
            self.bbox = [max(hmin-outline[0], 0), min(hmax+outline[1], H)+1,
                        max(wmin-outline[2], 0), min(wmax+outline[3], W)+1,
                        max(dmin-outline[4], 0), min(dmax+outline[5], D)+1]
            self.cropped_array = label_map[max(hmin-outline[0], 0): min(hmax+outline[1], H)+1,
                                        max(wmin-outline[2], 0): min(wmax+outline[3], W)+1,
                                        max(dmin-outline[4], 0): min(dmax+outline[5], D)+1]
            self.cropped_slice = [slice(max(hmin-outline[0], 0), min(hmax+outline[1], H)+1),
                                slice(max(wmin-outline[2], 0), min(wmax+outline[3], W)+1),
                                slice(max(dmin-outline[4], 0), min(dmax+outline[5], D)+1)]
        
        if return_slice:
            retval.append(self.cropped_slice)
        else:
            retval.append(self.bbox)
        
        if return_ndarray:
            retval.append(self.cropped_array)
            
        if len(retval) == 1:
            return retval[0]
        return retval
    
    def transform(self, point_or_points):
        # raw pt -> bbox pt
        point_or_points = np.array(point_or_points)
        if point_or_points.ndim == 1:
            point_or_points = point_or_points[np.newaxis]
        inv = point_or_points - np.asarray([self.bbox[0], self.bbox[2], self.bbox[4]])[np.newaxis]
        if np.any(inv < 0): return None
        if inv.shape[0] > 1:
            return inv
        else: return inv[0]
    
    def inv_transform(self, point_or_points):
        point_or_points = np.array(point_or_points)
        if point_or_points.ndim == 1:
            point_or_points = point_or_points[np.newaxis]
        trans = point_or_points + np.asarray([self.bbox[0], self.bbox[2], self.bbox[4]])[np.newaxis]
        if trans.shape[0] > 1:
            return trans
        return trans[0]
    
    @staticmethod
    def get_lil_box(label_map: np.ndarray, anchor: Iterable, outline: list, return_slice=False, return_ndarray=True):
        # this implementation does not transfer `outline` to its correct spacing
        # should pass a spacing-rectified `outline` when calling this function
        H, W, D = label_map.shape
        x, y, z = anchor
        if isinstance(outline, int):
            outline = [outline] * 6
        if len(outline) == 3:
            outline = reduce(lambda a, b: a+b, list(zip(outline, outline)))
            
        if return_slice:
            return [slice(max(x-outline[0], 0), min(x+outline[1], H)+1),
                    slice(max(y-outline[2], 0), min(y+outline[3], W)+1),
                    slice(max(z-outline[4], 0), min(z+outline[5], D)+1)]
        
        if return_ndarray:
            return label_map[max(x-outline[0], 0): min(x+outline[1], H)+1,
                            max(y-outline[2], 0): min(y+outline[3], W)+1,
                            max(z-outline[4], 0): min(z+outline[5], D)+1]
            
        return [max(x-outline[0], 0), min(x+outline[1], H)+1,
                max(y-outline[2], 0), min(y+outline[3], W)+1,
                max(z-outline[4], 0), min(z+outline[5], D)+1]

    
def find_largest_connected_components(mask, p=None, n=None, i0=0):
    if mask.sum() == 0:
        return mask
    labeled_mask, num_labels = label(mask)
    
    num_label_pixels = np.bincount(labeled_mask.flatten())
    num_label_pixels = [(l, num_label_pixels[l]) for l in range(1, num_labels+1)]
    label_nums = np.array(num_label_pixels)
    if p is not None:
        threshold = p * label_nums[:, 1].max()
        lcc = [_[0] for _ in num_label_pixels if _[1] >= threshold]
    elif n is not None:
        num_label_pixels.sort(reverse=True, key=lambda x: x[1])
        lcc = [num_label_pixels[i][0] for i in range(n)]
    else:
        lcc = [num_label_pixels[label_nums[:, 1].argmax()][0]]
    
    res = np.zeros(mask.shape, dtype=np.uint8)
    res_vein_art_label = [(labeled_index, np.argwhere(labeled_mask == labeled_index).mean(axis=0)) for labeled_index in lcc]
    if len(mask.shape) == 3:
        res_vein_art_label.sort(key=lambda x: x[1][2], reverse=True)  # sort from L to R
    for i, (labeled_index, _) in enumerate(res_vein_art_label):
        res[labeled_mask == labeled_index] = i0 + i + 1
    return res


def within_bounds(array, index_or_indices, assign=False, assign_value=None):
    index_or_indices = np.array(index_or_indices)
    cor_index_or_indices = np.zeros(index_or_indices.shape)
    
    if isinstance(index_or_indices, np.ndarray):
        if index_or_indices.ndim == 1:
            index_or_indices = index_or_indices[np.newaxis]
        
        _ = 0
        for _, index in enumerate(index_or_indices):
            if index.any() < 0 or np.any(index >= array.shape):
                continue
            if assign:
                array[tuple(index)] = assign_value
            cor_index_or_indices[_] = index
            
    return index_or_indices[:_]
    
        
def get_rot_xyz(rotXYZ):
        
        sines = np.sin(rotXYZ / 180 * np.pi)
        cosines = np.cos(rotXYZ / 180 * np.pi)
        rotX = np.asarray([[1, 0, 0], [0, cosines[0], -sines[0]], [0, sines[0], cosines[0]]])
        rotY = np.asarray([[cosines[1], 0, -sines[1]], [0, 1, 0], [sines[1], 0, cosines[1]]])
        rotZ = np.asarray([[cosines[2], -sines[2], 0], [sines[2], cosines[2], 0], [0, 0, 1]])
        rot = rotX @ rotY @ rotZ
        return rot


def get_dir(path, por_or_index):
    
    if len(path) <= 1:
        return None
    
    path = np.array(path)
    if isinstance(por_or_index, float): cutoff = round(len(path) * por_or_index)
    elif isinstance(por_or_index, int): cutoff = por_or_index if por_or_index > 0 else len(path) - por_or_index
    cutoff = np.clip(cutoff, 0, len(path) - 2)
    
    s = np.asarray([-(path[:cutoff] - path[cutoff]) * np.exp(-np.linspace(0, cutoff, cutoff))[::-1, np.newaxis]]).sum(axis=0)
    t = np.asarray([(path[cutoff+1:] - path[cutoff]) * np.exp(-np.linspace(0, len(path)-cutoff-1, len(path)-cutoff-1))[:, np.newaxis]]).sum(axis=0)
    direction = np.vstack([s, t]).mean(axis=0)
    direction /= np.linalg.norm(direction)
    return direction


def printf(msg, end='\n'):
    if msg.startswith("[INFO]"):
        print(f"proc <{os.getpid()}>", end=" ")
        print(colorama.Fore.GREEN, end='')
    elif msg.startswith("[WARNING]"):
        print(f"proc <{os.getpid()}>", end=" ")
        print(colorama.Fore.BLUE, end='')
    elif msg.startswith("[ERROR]"):
        print(f"proc <{os.getpid()}>", end=" ")
        print(colorama.Fore.RED, end='')
    elif msg.startswith("[DEBUG]"):
        print(f"proc <{os.getpid()}>", end=" ")
        print(colorama.Fore.YELLOW, end='')
    else:
        print(f"proc <{os.getpid()}>", end=" ")
    print(msg, end=end)
    print(colorama.Fore.WHITE, end='')
    return
    
    
def slice_orientation(array, code=None, percentile=50, orient=-1, direction=-1, half=-1):
    
    if code is not None:
        if code == 'L': orient, direction, half = (1, 2, 0)
        elif code == 'R': orient, direction, half = (1, 2, 1)
        elif code == "A": orient, direction, half = (2, 1, 0)
        elif code == 'P': orient, direction, half = (2, 1, 1)
        elif code == 'S': orient, direction, half = (2, 0, 0)
        elif code == 'I': orient, direction, half = (2, 0, 1)
    else: 
        if orient not in [0, 1, 2]: return array
    
    array = array.copy()
    axes = [0, 1, 2]
    axes.remove(orient)
    axis = np.argwhere(axes == orient)
    
    for layer in range(array.shape[orient]):
        if orient == 0:
            array_slice = array[layer]
        elif orient == 1:
            array_slice = array[:, layer]
        else:
            array_slice = array[:, :, layer]
        axis = 0 if direction == 0 else 1
        
        mean = np.argwhere(np.any(array_slice > 0, axis=1-axis))
        if len(mean) == 0: mean = 0 if half == 0 else -1
        else: mean = np.percentile(mean, percentile)
        retrieve = slice(0, round(mean)) if half == 0 else slice(round(mean), None)
        if axis == 0: array_slice[retrieve] = 0
        else: array_slice[:, retrieve] = 0
        
    return array
