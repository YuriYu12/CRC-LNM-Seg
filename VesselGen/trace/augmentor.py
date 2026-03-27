import os, sys
sys.path.append(os.getcwd())

import time
import shutil
import nibabel
import traceback
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

from os.path import *
from functools import wraps
from itertools import cycle
from collections import defaultdict
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt, binary_dilation

from VesselGen.trace.A_star import astar
from VesselGen.trace.distance_tracker import DTracker
from VesselGen.trace.postprocessor import PipelinePostprocessor
from VesselGen.macros import LABEL_MAPPING, ADJACENT_NODES_3D
from VesselGen.funcbase import Box, timeit,\
    find_largest_connected_components, get_dir, printf, slice_orientation

args = None

#   <front>                          <side>
#
#     ___          | S                 .         | S
#    /   \    L ---|--- R              |    A ---|--- P
#    \ . /         | I                .|         | I
#      |                               | 
#  ---------                           .
#      /\          | z-                |         | z-
#     /  \  x-  ---|--- x+             |   y+ ---|--- y-                            z   y   x
#    /    \        | z+                |         | z+        @sitk.Direction = diag(-1, -1, 1)


def listdir(path):
    return [join(path, subpath) for subpath in os.listdir(path) if not subpath.startswith('.')]


def makedir_or_dirs(path, destory_on_exist=False):
    if exists(path):
        if destory_on_exist:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path
        
        
def augment_totalseg(zipped_dirs, proc_id, expire=20):
    
    for itr, ct_path in enumerate(zipped_dirs):
        if itr > expire: break
        printf(f"************ {ct_path}, proc id {proc_id} ************")
        static_fields = dict(
            mode=-1,
            fname=ct_path.split('/')[-1],
            dirname='/'.join(ct_path.split('/')[:-1]),
        )
        augmentor = PipelineAugmentor(**static_fields)
        augmentor.generate_()
            

def augment_cmu(zipped_dirs, proc_id, expire=25):
    
    for itr, ct_path in enumerate(zipped_dirs):
        if itr > expire: break
        printf(f"************ {ct_path}, proc id {proc_id} ************")
        static_fields = dict(
            mode=-1,
            fname=ct_path.split('/')[-1],
            dirname='/'.join(ct_path.split('/')[:-1]),
        )
        augmentor = PipelineAugmentor(**static_fields)
        augmentor.generate_()


class PipelineAugmentor:
    R_VEIN_LABEL = 1
    R_ARTERY_LABEL = 2
    L_VEIN_LABEL = 4
    L_ARTERY_LABEL = 3
    # with LR AP IS rotation
    read_from_nifti_unrot = lambda filename, dtype=np.float32: nibabel.load(filename).dataobj[:].astype(dtype).transpose(2, 1, 0)
    
    def __init__(self, fname=None, dirname=None, mode=1, **func_fields):
        
        self.mode = mode
        self.skip = False
        self.fname = fname
        self.ct_dirname = dirname
        
        self._origin = None
        self._spacing = None
        self._direction = None
        
        self._load()
        self.__set_vessel_prob()
        self.mask[self.seg > 0] = 255  # seg primary vessels
        
        self.trace = dict()
        _ = lambda: defaultdict(_)
        self.start_point = _()
        self.end_point = _()
        self.bifur = _()
        self._astar = astar(**func_fields)
        
        self.foreground = np.zeros(self.mask.shape, dtype=np.uint8)
        self.background = None
        
    def __set_vessel_prob(self):
        self.sma, self.smv, self.ima, self.imv = [1] * 4
        self.ica, self.rca, self.mca = [1] * 3
        self.icv, self.rcv, self.srcv, self.mcv = [1] * 4
        self.lca, self.sa, self.sra = [1] * 3
        self.lcv, self.sv, self.srv = [1] * 3
        self.henle = self.icv + self.rcv + self.mcv + self.srcv
        self.veinx = self.lcv + self.sv + self.srv
        self.artx = self.lca + self.sa + self.sra + self.veinx
        self.veinx *= self.mode
        self.artx *= self.mode
        
    def __set_local_spacing(self, attr, force=False):
        if self._spacing is None: 
            self._spacing = attr
            printf(f"[INFO] SET nifti image spacing={self._spacing}")
        elif self._spacing != attr and not force: raise RuntimeError(f"spacing of read images are not the same, originally expected at {self._spacing}, got {attr}")
    
    def __set_local_direction(self, attr):
        if self._direction is None: 
            self._direction = attr
            printf(f"[INFO] SET nifti image direction={self._direction}")
        elif self._direction != attr: raise RuntimeError(f"direction of read images are not the same, originally tranformed at {self._direction}, got {attr}")
    
    def __set_local_origin(self, attr):
        if self._origin is None: 
            self._origin = attr
            printf(f"[INFO] SET nifti image origin={self._origin}")
        elif self._origin != attr: raise RuntimeError(f"origin of read images are not the same, originally tranformed at {self._origin}, got {attr}")
    
    origin = property(fget=lambda self: self._origin, fset=__set_local_origin)
    spacing = property(fget=lambda self: self._spacing, fset=__set_local_spacing)
    direction = property(fget=lambda self: self._direction, fset=__set_local_direction)
        
    def _load(self):
        image_path = join(os.environ.get("nnUNet_raw"), args.save_root, "imagesTr")
        mask_path = join(os.environ.get("nnUNet_raw"), args.save_root, "labelsTr")
        self.this_image_path = join(image_path, self.fname)   # Dataset000_XXXX_XXXX
        self.this_mask_path = join(mask_path, "_".join(self.fname.split("_")[:-1])) + ".nii.gz"  # Dataset000_XXXX
        
        if exists(self.this_image_path):
            self.skip = True
        
        self.ct = self._translate_array_nifti(nifti_or_niftis=join(self.ct_dirname, self.fname))
        self.mask = self._translate_array_nifti(nifti_or_niftis=join(self.ct_dirname.replace('image', 'mask'), self.fname))
        self.blank = self._translate_array_nifti(nifti_or_niftis=join(self.ct_dirname.replace('image', 'blank'), self.fname))
        
        self.seg = np.zeros(self.ct.shape, dtype=np.uint8)
        if self.mode > 0:
            seg_vessels = self._translate_array_nifti(nifti_or_niftis=join(self.ct_dirname.replace('image', 'seg'), self.fname))
            seg_focus_r = (seg_vessels == 1) | (seg_vessels == 8)
            seg_focus_r = find_largest_connected_components(seg_focus_r, n=2)  # get rid of noise
            seg_focus_l = (seg_vessels == 4)
            seg_focus_l = find_largest_connected_components(seg_focus_l, n=2, i0=2)  # get rid of noise
            # legacy
            """ori_pt = np.asarray(list(map(int, ct_cut.split()[0].split('_')[1:])))
            cut_pt = np.asarray(list(map(int, ct_cut.split()[1].rstrip('.nii.gz').split('_'))))
            offset = ori_pt - cut_pt
            
            self.seg[offset[0]: offset[0] + seg_focus_r.shape[0],
                    offset[1]: offset[1] + seg_focus_r.shape[1],
                    offset[2]: offset[2] + seg_focus_r.shape[2]] = seg_focus_r + seg_focus_l"""
            self.seg = seg_focus_r + seg_focus_l
        
    def _translate_array_nifti(self, ndarray_or_ndarrays=None, nifti_or_niftis=None, spacing=None):
        # TODO: accomodate nifti images of different direction and spacings
        #       act as a smarter variant of `funcbase.save_nifti()`
        
        if ndarray_or_ndarrays is not None:
            if spacing is None:
                spacing = self.spacing
            assert nifti_or_niftis is None
            if isinstance(ndarray_or_ndarrays, np.ndarray): ndarray_or_ndarrays = [ndarray_or_ndarrays]
            niftis = []
            for i_array, array in enumerate(ndarray_or_ndarrays):
                nifti_image = sitk.GetImageFromArray(array)
                nifti_image.SetDirection(np.diag(self.direction).flatten())
                nifti_image.SetOrigin(self.origin)
                nifti_image.SetSpacing(spacing[::-1])
                niftis.append(nifti_image)
            return niftis[0] if len(niftis) == 1 else niftis
        
        if nifti_or_niftis is not None:
            assert ndarray_or_ndarrays is None
            if isinstance(nifti_or_niftis, (str, sitk.Image)): nifti_or_niftis = [nifti_or_niftis]
            arrays = []
            for i_nifti, nifti in enumerate(nifti_or_niftis):
                if isinstance(nifti, str): nifti = sitk.ReadImage(nifti)
                if i_nifti == 0:
                    self.origin = nifti.GetOrigin()
                    self.spacing = nifti.GetSpacing()[::-1]
                    self.direction = tuple(np.diag(np.asarray(nifti.GetDirection()).reshape(3, 3)))

                numpy_array = sitk.GetArrayFromImage(nifti)
                arrays.append(numpy_array)
                
            if i_nifti == 0:
                self.X = lambda mm, T=round: T(mm / self.spacing[0])
                self.Y = lambda mm, T=round: T(mm / self.spacing[1])
                self.Z = lambda mm, T=round: T(mm / self.spacing[2])
                self.XYZ_AVG = lambda mm, T=round: T((self.X(mm, T) + self.Y(mm, T) + self.Z(mm, T)) / 3)
            return arrays[0] if len(arrays) == 1 else arrays
    
    def join(self, *args, **kwargs):
        _dict = defaultdict(list)
        _add = kwargs.get('add', 0)
        for _vessel in args:
            for _bifur in _vessel.keys():
                for __vessel in _vessel[_bifur]:
                    if len(__vessel) > 0:
                        _dict[_bifur + _add].append(__vessel)
        return _dict
    
    def exception_catcher(total_itrs=4):
        exception_stack = []
        def wrapper(func):
            @wraps(func)
            def catcher(self, *args, **kwargs):
                ret = None
                printf(f"[INFO] running {func.__name__}")
                for _ in range(total_itrs):
                    try:
                        ret = func(self, *args, **kwargs)
                        break
                    except Exception as e:
                        exception_stack.append(e)
                        for vessel_name in self.trace:
                            self.__dict__[vessel_name] = 0
                        printf(f"[ERROR] the following exception raised, retrying {len(exception_stack)} / {total_itrs} ...")
                        traceback.print_exception(e)
                else:
                    try:
                        ret = func(self, *args, **kwargs)
                    except Exception as e:
                        traceback.print_exception(e)
                        return
                return ret
            return catcher
        return wrapper
    
    def _register(self, vessel_name, vessel_path):
        if self.trace.__contains__(vessel_name): self.trace[vessel_name] = self.join(self.trace[vessel_name], vessel_path)
        else: self.trace[vessel_name] = vessel_path
        
    def _collect(self, *vessel_names):
        if len(vessel_names) == 0:
            vessel_names = list(self.trace.keys())
        return self.join(*[self.trace[v] for v in vessel_names])
    
    @timeit()
    @exception_catcher(total_itrs=4)
    def generate_(self):
        
        if self.skip: 
            printf(f"[INFO] required fore-background pair {self.this_image_path} exists")
            return
        
        if self.mode < 0:
            self.seg[...] = 0
            with timeit("primary vessel timer"):
                # sma
                if np.random.random() < self.sma:
                    printf("GENERATING SMA ...")
                    vert_l1_mask_z = np.argwhere(self.mask == LABEL_MAPPING['vertebrae_L1']).mean(axis=0)[0]
                    sma_start_section = np.asarray([_ for _ in np.argwhere(self.mask == LABEL_MAPPING['aorta']) if abs(_[0] - vert_l1_mask_z) < self.XYZ_AVG(20)])
                    sma_start_section = np.asarray([_ for _ in sma_start_section if _[1] > np.percentile(sma_start_section[:, 1], 80)])
                    sma_start_point = sma_start_section[np.random.randint(len(sma_start_section) - 1)]
                    sma_nearest_pan_pts = np.asarray([_ for _ in np.argwhere(self.mask == LABEL_MAPPING['pancreas'])])
                    sma_nearest_pan_pt = sma_nearest_pan_pts[np.asarray([np.linalg.norm(_ - np.asarray(sma_start_point)) for _ in sma_nearest_pan_pts]).argmin()]
                    sma0_end_point = sma_nearest_pan_pt + np.asarray([np.random.randint(self.Z(-7), self.Z(-3)), np.random.randint(self.Y(-7), self.Y(-3)), np.random.randint(self.X(-2), self.X(2))])
                    duo_mask_zmin = round(np.percentile(np.argwhere(self.mask == LABEL_MAPPING['duodenum']), 20, axis=0)[0])
                    sma1_end_point = np.asarray([duo_mask_zmin, sma0_end_point[1], sma0_end_point[2]+np.random.randint(self.X(-3), 0)])
                    
                    self.start_point['sma'] = sma_start_point
                    self.end_point['sma'] = sma1_end_point
                    self.bifur['sma'] = None
                    
                    self.run('sma', level=9, search_span=100, avoid_span=4, start_dir=(-1, 1, 0))

                # smv
                if np.random.random() < self.smv:
                    printf("GENERATING SMV ...")
                    smv_nearest_port_pts = np.asarray([_ for _ in np.argwhere(self.mask == LABEL_MAPPING['portal_vein_and_splenic_vein'])])
                    smv_nearest_port_pts = np.asarray([_ for _ in smv_nearest_port_pts if _[0] < np.percentile(smv_nearest_port_pts[:, 0], 20)])
                    smv_nearest_port_pt = smv_nearest_port_pts[np.asarray([np.linalg.norm(_ - np.asarray(sma_start_point)) for _ in smv_nearest_port_pts]).argmin()]
                    smv_start_point = smv_nearest_port_pt + np.asarray([0, 0, np.random.randint(self.X(15), self.X(25))])
                    smv_end_point = sma1_end_point + np.asarray([np.random.randint(self.Z(-5), self.Z(5)), np.random.randint(self.Y(-5), self.Y(5)), np.random.randint(self.X(8), self.X(12))])
                    
                    self.start_point['smv'] = smv_start_point
                    self.end_point['smv'] = smv_end_point
                    self.bifur['smv'] = None
                    
                    self.run('smv', level=19, search_span=100, avoid_span=4, start_dir=(-1, 1, 0))    
                
                # ima
                if np.random.random() < self.ima:
                    printf("GENERATING IMA ...")
                    vert_l3_mask_z = np.argwhere(self.mask == LABEL_MAPPING['vertebrae_L3']).mean(axis=0)[0]
                    ima_start_section = np.asarray([_ for _ in np.argwhere(self.mask == LABEL_MAPPING['aorta']) if abs(_[0] - vert_l3_mask_z) < self.XYZ_AVG(20)])
                    ima_start_section = np.asarray([_ for _ in ima_start_section if _[1] > np.percentile(ima_start_section[:, 1], 80)])
                    ima_start_point = ima_start_section[np.random.randint(len(ima_start_section))]
                    ima_end_section = np.argwhere(self.mask == LABEL_MAPPING['iliac_artery_left'])
                    ima_end_section = [_ for _ in ima_end_section if self.Z(20) < abs(_[0] - ima_end_section[:, 0].max()) < self.Z(40)]
                    rand_offset = np.asarray([np.random.randint(self.Z(10), self.Z(20)), np.random.randint(self.Y(2), self.Y(5)), np.random.randint(self.X(10), self.X(20))])
                    ima_end_point = ima_end_section[np.random.randint(len(ima_end_section))] + rand_offset * np.asarray([np.random.choice([-1, 1]), 0, np.random.choice([-1, 1])])
                    
                    self.start_point['ima'] = ima_start_point
                    self.end_point['ima'] = ima_end_point
                    self.bifur['ima'] = None
                    
                    self.run('ima', level=29, search_span=100, avoid_span=5, cutoff=(72, 88), start_dir=(-1, 1, 0), exit_step=5000)  # maybe cutoff + 30 to avoid to short generation
                
                # imv
                if np.random.random() < self.imv:
                    printf("GENERATING IMV ...")
                    smv_start_point = self.trace['smv'][19][0][0]
                    imv_start_section = np.asarray([_ for _ in np.argwhere(self.mask == LABEL_MAPPING['portal_vein_and_splenic_vein']) if self.X(30) < _[2] - smv_start_point[2] < self.X(100)])
                    imv_start_section = [_ for _ in imv_start_section if _[0] < np.percentile(imv_start_section[:, 0], 20)]
                    imv_start_point = imv_start_section[np.random.randint(len(imv_start_section))]
                    imv_end_point = np.array(self.trace['ima'][29][0][-1]) + np.asarray([np.random.randint(self.Z(-10), self.Z(10)), np.random.randint(self.Y(-5), self.Y(5)), np.random.randint(self.X(5), self.X(20))])
                    
                    self.start_point['imv'] = imv_start_point
                    self.end_point['imv'] = imv_end_point
                    self.bifur['imv'] = None
                    
                    self.run('imv', level=39, search_span=100, avoid_span=4, start_dir=(-1, 1, 0), exit_step=5000)
        else:
            self.trace['sma'] = {9: [np.argwhere(skeletonize_3d(self.seg == self.R_ARTERY_LABEL)).tolist()]}
            self.trace['smv'] = {19: [np.argwhere(skeletonize_3d(self.seg == self.R_VEIN_LABEL)).tolist()]}
            self.trace['ima'] = {29: [np.argwhere(skeletonize_3d(self.seg == self.L_ARTERY_LABEL)).tolist()]}
            self.trace['imv'] = {39: [np.argwhere(skeletonize_3d(self.seg == self.L_VEIN_LABEL)).tolist()]}
                    
        bbox = Box()
        a_mask = self.mask == LABEL_MAPPING['ascendant_colon']
        t_mask = self.mask == LABEL_MAPPING['transversal_colon']
        a_mask[slice_orientation(a_mask, code='R') > 0] = False
        t_mask[slice_orientation(t_mask, code='A') > 0] = False
        a_masked_pts = np.argwhere(a_mask)
        t_masked_pts = np.argwhere(t_mask)
        
        _, _ = bbox(a_mask, outline=5, return_slice=True, return_ndarray=True)
        
        sma_skl = np.array(self.trace['sma'][9][0])
        # 1) ICA
        if np.random.random() < self.ica:
            printf('GENERATING ICA ...')
            visited = np.zeros(a_mask.shape, dtype=np.uint8)
            br_pt = sorted(a_masked_pts.tolist(), key=lambda x: self.Z(x[0]) - 1.5 * self.X(x[2]))[0]
            sma_lower_lmt = tuple(np.mean([_ for _ in sma_skl if _[0] == sma_skl[:, 0].min()], dtype=int, axis=0))
            self.start_point['ica'] = sma_lower_lmt

            # refine to zmin
            candidate = [br_pt]
            while len(candidate) > 0:
                br_pt = candidate.pop(0)
                for delta in [_ for _ in ADJACENT_NODES_3D if _[0] <= 0]:
                    current_pt = tuple(br_pt[i] + delta[i] for i in range(3))
                    if visited[current_pt] > 0:
                        continue
                    elif a_mask[current_pt]:
                        index = 0
                        visited[current_pt] = 1
                        # insert sort (dfs, priorize z)
                        while index < len(candidate) and candidate[index][0] > br_pt[0] + delta[0]:
                            index += 1
                        candidate.insert(index, current_pt)

            self.end_point['ica'] = br_pt
            self.bifur['ica'] = {0: {"p": self.XYZ_AVG(mm=30), "n": (3, 7), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('ica', poi=a_mask)
        self.ica = 0
            
        # 2) RCA
        if np.random.random() < self.rca:
            printf('GENERATING RCA ...')
            rnd = np.random.random()
            if rnd < .708:
                z, y, x = self.start_point['ica']
                rca_start_point = np.asarray([z + self.Z(mm=16), y, x])
                self.start_point['rca'] = rca_start_point
            elif rnd < .846:
                self.start_point['rca'] = self.trace['ica'][0][0][np.random.choice(np.arange(len(self.trace['ica'][0][0])))]
            if rnd < .846:
                coord = [_ for _ in a_masked_pts if np.abs(_[0] - self.start_point['rca'][0]) < self.Z(mm=5)]
                self.end_point['rca'] = coord[np.random.randint(len(coord))]
                self.bifur['rca'] = {0: {"p": self.XYZ_AVG(mm=30), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
                self.run('rca', poi=a_mask)
        self.rca = 0
        
        # 3) MCA
        if np.random.random() < self.mca:
            printf('GENERATING MCA ...')
            rnd = np.random.random()
            rnd2 = np.random.random()
            sma_lower_lmt = tuple(np.mean([_ for _ in sma_skl if _[0] == sma_skl[:, 0].min()], dtype=int, axis=0))
            if rnd2 < .884:
                if rnd < .787:
                    sma_upper_lmt = tuple(np.mean([_ for _ in sma_skl if _[0] == sma_skl[:, 0].max()], dtype=int, axis=0))
                    start_pt = [_ for _ in sma_skl if _[0] < sma_upper_lmt[0] and _[0] > sma_lower_lmt[0]]
                    self.start_point['mca'][0] = start_pt[np.random.choice(np.arange(len(start_pt)))]
                elif rnd < .965 and self.trace.__contains__('rca'):
                    self.start_point['mca'][0] = self.trace['rca'][0][0][np.random.choice(np.arange(len(self.trace['rca'][0][0])))]
                    
                if rnd < .787 or (rnd < .965 and self.trace.__contains__('rca')):
                    mca2t_nearest_pt = t_masked_pts[np.linalg.norm(t_masked_pts - self.start_point['mca'][0]).argmin()]
                    third_quantile_mca2t = self.start_point['mca'][0] + (np.random.random() * 2/3 + 1/3) * (mca2t_nearest_pt[-1] - self.start_point['mca'][0])
                    self.end_point['mca'][0] = np.round(third_quantile_mca2t).astype(int)
                    self.bifur['mca'][0] = None
                    self.run("mca", 0, poi=t_mask)
                    # L-MCA
                    self.start_point['mca'][1] = self.end_point['mca'][0]
                    end_pt = [_ for _ in t_masked_pts if _[2] < t_masked_pts[:, 2].mean()]
                    self.end_point['mca'][1] = end_pt[np.random.randint(len(end_pt))]
                    self.bifur['mca'][1] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=30), "n": (1, 3), "b": None}}}}
                    self.run("mca", 1, poi=t_mask)
                    # R-MCA
                    self.start_point['mca'][1] = self.end_point['mca'][0]
                    end_pt = [_ for _ in t_masked_pts if _[2] > t_masked_pts[:, 2].mean()] 
                    self.end_point['mca'][1] = end_pt[np.random.randint(len(end_pt))]
                    self.bifur['mca'][1] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=30), "n": (1, 3), "b": None}}}}
                    self.run("mca", 1, poi=t_mask)
                    
            if rnd2 < 1:
                start_points = []
                if rnd < .787:
                    sma_upper_lmt = tuple(np.mean([_ for _ in sma_skl if _[0] == sma_skl[:, 0].max()], dtype=int, axis=0))
                    start_pt = [_ for _ in sma_skl if _[0] < sma_upper_lmt[0] and _[0] > sma_lower_lmt[0]]
                    start_points.append(start_pt[np.random.choice(np.arange(len(start_pt)))])
                    start_pt = [_ for _ in start_pt if np.linalg.norm(_ - np.array(start_points[0])) > self.XYZ_AVG(5)]
                    start_points.append(start_pt[np.random.choice(np.arange(len(start_pt)))])
                elif rnd < .965 and self.trace.__contains__('rca'):
                    rca_main = self.trace['rca'][0][0]
                    rca_choices = np.random.choice(np.arange(len(rca_main)), size=2)
                    start_points.append(rca_main[rca_choices[0]])
                    start_points.append(rca_main[rca_choices[1]])
                
                if rnd < .787 or (rnd < .965 and self.trace.__contains__('rca')):
                    # L-MCA
                    self.start_point['mca'] = start_points[0]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] < t_masked_pts[:, 2].mean()])
                    self.end_point['mca'] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mca'] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=30), "n": (1, 3), "b": None}}}}
                    self.run("mca", poi=t_mask)
                    # R-MCA
                    self.start_point['mca'] = start_points[1]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] > t_masked_pts[:, 2].mean()])
                    self.end_point['mca'] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mca'] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=30), "n": (1, 3), "b": None}}}}
                    self.run("mca", poi=t_mask)
        self.mca = 0
            
        smv_skl = np.array(self.trace['smv'][19][0])
        # 4) ICV
        if np.random.random() < self.icv:
            printf('GENERATING ICV ...')
            br_pt = sorted(a_masked_pts.tolist(), key=lambda x: self.Z(x[0]) - 1.5 * self.X(x[2]))[0]
            smv_lower_lmt = tuple(np.mean([_ for _ in smv_skl if _[0] == smv_skl[:, 0].min()], dtype=int, axis=0))
            self.start_point['icv'] = smv_lower_lmt
            self.end_point['icv'] = br_pt
            self.bifur['icv'] = {0: {"p": self.XYZ_AVG(mm=30), "n": (3, 7), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('icv', level=10, poi=a_mask)
        self.icv = 0
            
        # 5) Henle
        if self.henle:
            printf('GENERATING Henle ...')
            smv_upper_lmt = tuple(np.mean([_ for _ in smv_skl if _[0] == smv_skl[:, 0].max()], dtype=int, axis=0))
            z, y, x = smv_upper_lmt
            z_ = z + self.Z(mm=np.random.randint(5, 9))
            self.start_point['henle'] = np.asarray((z_, y, x))
            self.end_point['henle'] = np.asarray(a_masked_pts[np.linalg.norm(a_masked_pts - self.start_point['henle']).argmin()])
            self.bifur['henle'] = None
            
            self.run('henle', level=10, cutoff=(70, 151))
            henle = self.trace['henle'][10][0]
            
        # 6) RCV
        if np.random.random() < self.rcv:
            printf('GENERATING RCV ...')
            rnd = np.random.random()
            if rnd < .49:
                z, y, x = self.start_point['icv']
                smv_lower_lmt = np.asarray([z + self.Z(mm=16), y, x])
                self.start_point['rcv'] = smv_lower_lmt
            elif rnd < .993:
                henle = self.trace['henle'][10][0]
                self.start_point['rcv'] = henle[np.random.choice(np.arange(len(henle)))]
                henle = [_ for _ in henle if np.linalg.norm(_ - np.array(self.start_point['rcv'])) > self.XYZ_AVG(5)]
            if rnd < .993:
                coord = [_ for _ in a_masked_pts if np.abs(_[0] - self.start_point['rcv'][0]) < self.Z(mm=5)]
                self.end_point['rcv'] = coord[np.random.randint(len(coord))]
                self.bifur['rcv'] = {0: {"p": self.XYZ_AVG(mm=30), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
                self.run('rcv', level=10, poi=a_mask)
        self.rcv = 0
                
        # 7) SRCV
        if np.random.random() < self.srcv:
            printf('GENERATING SRCV ...')
            henle = self.trace['henle'][10][0]
            visited = np.zeros(a_mask.shape, dtype=np.uint8)
            br_pt = sorted(a_masked_pts.tolist(), key=lambda x: - self.Z(x[0]) - self.X(x[2]))[0]
            
            # refine to zmin
            candidate = [br_pt]
            while len(candidate) > 0:
                br_pt = candidate.pop(0)
                for delta in [_ for _ in ADJACENT_NODES_3D if _[0] <= 0]:
                    current_pt = tuple(br_pt[i] + delta[i] for i in range(3))
                    if visited[current_pt] > 0:
                        continue
                    elif a_mask[current_pt]:
                        index = 0
                        visited[current_pt] = 1
                        # insert sort (dfs, priorize z)
                        while index < len(candidate) and candidate[index][0] > br_pt[0] + delta[0]:
                            index += 1
                        candidate.insert(index, current_pt)
                        
            self.start_point['srcv'] = henle[np.random.randint(len(henle))]
            henle = [_ for _ in henle if np.linalg.norm(_ - np.array(self.start_point['srcv'])) > self.XYZ_AVG(5)]
            self.end_point['srcv'] = br_pt
            self.bifur['srcv'] = {0: {"p": self.XYZ_AVG(mm=30), "n": (1, 3), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('srcv', level=10, poi=a_mask)
        self.srcv = 0
            
        # 8) MCV
        if np.random.random() < self.mcv:
            printf('GENERATING MCV ...')
            rnd = np.random.random()
            rnd2 = np.random.random()
            henle = self.trace['henle'][10][0]
            if rnd2 < .697:
                if rnd < .832:
                    start_pt = [_ for _ in smv_skl if np.linalg.norm(_ - self.trace['henle'][10][0][0]) < self.XYZ_AVG(50)]
                    self.start_point['mcv'][0] = start_pt[np.random.randint(len(start_pt))]
                elif rnd < .946:
                    self.start_point['mcv'][0] = henle[np.random.randint(len(henle))]
                    henle = [_ for _ in henle if np.linalg.norm(_ - np.array(self.start_point['mcv'][0])) > self.XYZ_AVG(5)]
                
                if rnd < .946:
                    mcv2t_nearest_pt = t_masked_pts[np.linalg.norm(t_masked_pts - self.start_point['mcv'][0]).argmin()]
                    third_quantile_mcv2t = self.start_point['mcv'][0] + (np.random.random() * 2/3 + 1/3) * (mcv2t_nearest_pt[-1] - self.start_point['mcv'][0])
                    self.end_point['mcv'][0] = np.round(third_quantile_mcv2t).astype(int)
                    self.bifur['mcv'][0] = None
                    self.run("mcv", 0, level=10, poi=t_mask)
                    # L-MCV
                    self.start_point['mcv'][1] = self.end_point['mcv'][0]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] < t_masked_pts[:, 2].mean()])
                    self.end_point['mcv'][1] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mcv'][1] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (1, 3), "b": None}}}}
                    self.run("mcv", 1, level=10, poi=t_mask)
                    # R-MCV
                    self.start_point['mcv'][1] = self.end_point['mcv'][0]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] > t_masked_pts[:, 2].mean()])
                    self.end_point['mcv'][1] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mcv'][1] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (1, 3), "b": None}}}}
                    self.run("mcv", 1, level=10, poi=t_mask)
                    
            elif rnd2 < .956:
                start_points = []
                if rnd < .832:
                    start_pt = [_ for _ in smv_skl if np.linalg.norm(_ - self.trace['henle'][10][0][0]) < self.XYZ_AVG(50)]
                    start_points.append(start_pt[np.random.choice(np.arange(len(start_pt)))])
                    start_pt = [_ for _ in start_pt if np.linalg.norm(_ - np.array(start_points[0])) > self.XYZ_AVG(5)]
                    start_points.append(start_pt[np.random.choice(np.arange(len(start_pt)))])
                elif rnd < .946:
                    start_points.append(henle[np.random.choice(np.arange(len(henle)))])
                    henle = [_ for _ in henle if np.linalg.norm(_ - np.array(start_points[0])) > self.XYZ_AVG(5)]
                    start_points.append(henle[np.random.choice(np.arange(len(henle)))])
                
                if rnd < .946:
                    # L-MCV
                    self.start_point['mcv'] = start_points[0]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] < t_masked_pts[:, 2].mean()])
                    self.end_point['mcv'] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mcv'] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (1, 3), "b": None}}}}
                    self.run("mcv", level=10, poi=t_mask)
                    # R-MCV
                    self.start_point['mcv'] = start_points[1]
                    end_pt = np.asarray([_ for _ in t_masked_pts if _[2] > t_masked_pts[:, 2].mean()])
                    self.end_point['mcv'] = np.round(end_pt.mean(axis=0)).astype(int)
                    self.bifur['mcv'] = {0: {"p": self.XYZ_AVG(mm=40), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (1, 3), "b": None}}}}
                    self.run("mcv", level=10, poi=t_mask)
        self.mcv = 0
                    
        d_mask = (self.mask == LABEL_MAPPING['descendant_colon'])
        s_mask = (self.mask == LABEL_MAPPING['sigmoid_colon'])
        d_mask[slice_orientation(d_mask, code='L') > 0] = False
        s_mask[slice_orientation(s_mask, code='I') > 0] = False
        d_masked_pts = np.argwhere(d_mask)
        s_masked_pts = np.argwhere(s_mask)
        
        ima_skl = np.array(self.trace['ima'][29][0])
        # 9) ArtX
        if self.artx > 0:
            printf('GENERATING ArtX ...')
            ima_lower_lmt = tuple(np.mean([_ for _ in ima_skl if _[0] == ima_skl[:, 0].min()], dtype=int, axis=0))
            ima_len = (np.abs(ima_skl[0] - ima_skl[-1]) * np.array(self.spacing)).sum() * .75  # compensation
            ima_direction = -get_dir(ima_skl, -1)
            
            _artx_start_pt = ima_lower_lmt
            _artx_end_pt = np.round(np.array(_artx_start_pt + ima_direction * np.random.randint(60, 70))).astype(int)
            _trace = self._astar(self.mask, _artx_start_pt, _artx_end_pt, start_dir=ima_direction)
            _artx_len = (np.abs(np.array(_artx_start_pt) - np.array(_artx_end_pt)) * np.array(self.spacing)).sum()
            artx = _trace[:round(len(_trace) * (np.random.randint(42, 58) - ima_len) / _artx_len)]
            _artx = _trace[:round(len(_trace) * (np.random.randint(42, 58) - ima_len) / _artx_len)]
            for pt in artx: self.mask[*pt] = 255
            
        elif self.artx < 0:
            artx = ima_skl[:]
            printf(f"[DEBUG] tracked artx length {np.sum([np.abs(np.array(artx[i+1]) - np.array(artx[i])) for i in range(len(artx)-1)], axis=0).dot(np.array(self.spacing))} of {len(artx)} points")
        
        # 10) SA
        if np.random.random() < self.sa:
            printf('GENERATING SA ...')
            rnd = np.random.randint(1, 4)
            artx = ima_skl[:]
            coord = s_masked_pts.copy()
            while rnd > 0:
                self.start_point['sa'] = artx[np.random.randint(len(artx))]
                artx = [_ for _ in artx if np.linalg.norm(_ - np.array(self.start_point['sa'])) > self.XYZ_AVG(3)]
                self.end_point['sa'] = coord[np.random.randint(len(coord))]
                coord = [_ for _ in coord if np.linalg.norm(_ - np.array(self.end_point['sa'])) > self.XYZ_AVG(3)]
                self.bifur['sa'] = {0: {"p": self.XYZ_AVG(mm=50), "n": (2, 4), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 4), "b": None}}}}
                self.run('sa', poi=s_mask, level=20)
                rnd -= 1
        self.sa = 0
            
        # 11) LCA
        if np.random.random() < self.lca:
            printf('GENERATING LCA ...')
            rnd = np.random.random()
            if rnd < .473:
                ima_lower_lmt = tuple(np.mean([_ for _ in ima_skl if _[0] == ima_skl[:, 0].min()], dtype=int, axis=0))
                self.start_point['lca'] = ima_lower_lmt
            elif rnd < .952:
                sa_start = [_pt for _vessel in self.trace['sa'][20][0] for _pt in _vessel if np.linalg.norm(np.array(_pt) - np.array(_vessel[0])) < self.XYZ_AVG(30)]
                self.start_point['lca'] = sa_start[np.random.choice(np.arange(len(sa_start)))]

            if rnd < .952:
                tl_pt = sorted(d_masked_pts.tolist(), key=lambda x: -self.Z(x[0]) + 1.5 * self.X(x[2]))[0]
                self.bifur['lca'] = {0: {"p": self.XYZ_AVG(mm=50), "n": (3, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (1, 3), "b": None}}}}
                self.end_point['lca'] = tl_pt
                self.run('lca', poi=d_mask, level=20)
        self.lca = 0
                
        imv_skl = np.array(self.trace['imv'][39][0])
        # 12) VeinX
        if self.veinx > 0:
            printf('GENERATING VeinX ...')
            imv_lower_lmt = tuple(np.mean([_ for _ in imv_skl if _[0] == imv_skl[:, 0].min()], dtype=int, axis=0))
            _veinx_start_pt = imv_lower_lmt
            _veinx_end_pt = (_artx_end_pt[0] + np.random.randint(-self.Z(10), self.Z(10)),
                            _artx_end_pt[1] + np.random.randint(-self.Y(10), self.Y(10)),
                            _artx_end_pt[2] + np.random.randint(-self.X(10), self.X(10)))
            _trace = self._astar(self.mask, _veinx_start_pt, _veinx_end_pt)
            veinx = _trace[:]
            for pt in veinx: self.mask[*pt] = 255
            
        elif self.veinx < 0:
            _imv_len = (np.abs(np.array(imv_skl[0]) - np.array(imv_skl[-1])) * np.array(self.spacing)).sum()
            veinx = imv_skl[-round(np.random.randint(20, 50) / _imv_len * len(imv_skl)):]
            printf(f"[DEBUG] tracked veinx length {np.sum([np.abs(np.array(veinx[i+1]) - np.array(veinx[i])) for i in range(len(veinx)-1)], axis=0).dot(np.array(self.spacing))} of {len(veinx)} points")
        
        # 13) SV
        if np.random.random() < self.sv:
            printf('GENERATING SV ...')
            rnd = np.random.randint(1, 4)
            coord = s_masked_pts.copy()
            veinx = imv_skl[-round(np.random.randint(20, 50) / _imv_len * len(imv_skl)):]
            while rnd > 0:
                self.start_point['sv'] = veinx[np.random.randint(len(veinx))]
                veinx = [_ for _ in veinx if np.linalg.norm(_ - np.array(self.start_point['sv'])) > self.XYZ_AVG(3)]
                self.end_point['sv'] = coord[np.random.randint(len(coord))]
                coord = [_ for _ in coord if np.linalg.norm(_ - np.array(self.end_point['sv'])) > self.XYZ_AVG(3)]
                self.bifur['sv'] = {0: {"p": self.XYZ_AVG(mm=50), "n": (2, 4), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 4), "b": None}}}}
                self.run('sv', level=30, poi=s_mask)
                rnd -= 1
        self.sv = 0
            
        # 14) LCV
        if np.random.random() < self.lcv:
            printf('GENERATING LCV ...')
            rnd = np.random.random()
            if rnd < .5:
                imv_lower_lmt = tuple(np.mean([_ for _ in imv_skl if _[0] == imv_skl[:, 0].min()], dtype=int, axis=0))
                self.start_point['lcv'] = imv_lower_lmt
            else:
                sv_start = [_pt for _vessel in self.trace['sv'][30][0] for _pt in _vessel if np.linalg.norm(np.array(_pt) - np.array(_vessel[0])) < self.XYZ_AVG(30)]
                self.start_point['lcv'] = sv_start[np.random.choice(np.arange(len(sv_start)))]
                
            self.bifur['lcv'] = {0: {"p": self.XYZ_AVG(mm=50), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            tl_pt = sorted(d_masked_pts.tolist(), key=lambda x: -self.Z(x[0]) + 1.5 * self.X(x[2]))[0]
            self.end_point['lcv'] = tl_pt
            self.run('lcv', level=30, poi=d_mask)
        self.lcv = 0
            
        r_mask = self.mask == LABEL_MAPPING['rectum']
        r_mask[slice_orientation(r_mask, code='A') > 0] = False
        r_masked_pts = np.argwhere(r_mask)
        
        # 15) SRA
        if np.random.random() < self.sra:
            printf('GENERATING SRA ...')
            artx = ima_skl[:]
            artx_final_pt = artx[-1]
            # L-SRA
            # lcos = r_colon_grad.transpose(1, 2, 3, 0).dot(np.asarray([0, 0, -1]))
            il_pt = sorted(r_masked_pts.tolist(), key=lambda x: self.Y(x[1]) + self.Z(x[0]))[0]
            self.start_point['sra'] = artx_final_pt
            self.end_point['sra'] = il_pt
            self.bifur['sra'] = {0: {"p": self.XYZ_AVG(mm=55), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('sra', poi=r_mask, level=20, avoid_span=3, exit_step=5000, start_dir=np.asarray([-1, 1, 0]), Lambda=0.92)
            # R-SRA
            # rcos = r_colon_grad.transpose(1, 2, 3, 0).dot(np.asarray([0, 0, 1]))
            ir_pt = sorted(r_masked_pts.tolist(), key=lambda x: -self.Y(x[1]) + self.Z(x[0]))[0]
            self.start_point['sra'] = artx_final_pt
            self.end_point['sra'] = ir_pt
            self.bifur['sra'] = {0: {"p": self.XYZ_AVG(mm=55), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('sra', poi=r_mask, level=20, avoid_span=3, exit_step=5000, start_dir=np.asarray([-1, 1, 0]), Lambda=0.92)
        self.sra = 0
                
        # 16) SRV
        if np.random.random() < self.srv:
            printf('GENERATING SRV ...')
            veinx = imv_skl[-round(np.random.randint(20, 50) / _imv_len * len(imv_skl)):]
            veinx_final_pt = veinx[-1]
            # L-SRV
            il_pt = sorted(r_masked_pts.tolist(), key=lambda x: self.Y(x[1]) + self.Z(x[0]))[0]
            self.start_point['srv'] = veinx_final_pt
            self.end_point['srv'] = il_pt
            self.bifur['srv'] = {0: {"p": self.XYZ_AVG(mm=55), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('srv', poi=r_mask, level=30, avoid_span=3, exit_step=5000, start_dir=np.asarray([-1, 1, 0]), Lambda=0.92)
            # R-SRV
            ir_pt = sorted(r_masked_pts.tolist(), key=lambda x: -self.Y(x[1]) + self.Z(x[0]))[0]
            self.start_point['srv'] = veinx_final_pt
            self.end_point['srv'] = ir_pt
            self.bifur['srv'] = {0: {"p": self.XYZ_AVG(mm=55), "n": (2, 5), "b": {0: {"p": self.XYZ_AVG(mm=40), "n": (0, 3), "b": None}}}}
            self.run('srv', poi=r_mask, level=30, avoid_span=3, exit_step=5000, start_dir=np.asarray([-1, 1, 0]), Lambda=0.92)
        self.srv = 0
        
        printf(f"GENERATED VESSEL(s) {', '.join(self.trace.keys())}, PLOTTING ...")
        with timeit('local postprocess timer'):
            self.save_()
        return
    
    def save_(self, code=0):
        
        self.foreground[(self.mask >= 105) & (self.mask <= 109)] = self.mask[(self.mask >= 105) & (self.mask <= 109)]
        self.foreground[self.mask == LABEL_MAPPING["aorta"]] = 50
        self.foreground[self.mask == LABEL_MAPPING["portal_vein_and_splenic_vein"]] = 50
        self.background = self.blank
        foreground_im = self._translate_array_nifti(ndarray_or_ndarrays=self.foreground, spacing=self.spacing)
        background_im = self._translate_array_nifti(ndarray_or_ndarrays=self.background, spacing=self.spacing)
        
        printf(f'[INFO] saving generated foreground and background to local ...')
        sitk.WriteImage(foreground_im, self.this_mask_path)
        sitk.WriteImage(background_im, self.this_image_path)
        printf(f"[INFO] saved generated foreground and background at {self.this_image_path.replace('image', '*****')}")
        
    def run(self, *vessel_name, **kwargs):
        
        vessel_level = kwargs.get('level', 0)
        
        def get_from_local_dict(dict_name, key):
            nonlocal vessel_level
            k_of_key = 0
            while k_of_key < len(key):
                dict_name = dict_name[key[k_of_key]]
                k_of_key += 1
            vessel_level += k_of_key - 1
            return dict_name
        
        func_dict = dict(
            XYZ_AVG = self.XYZ_AVG,
            spacing = self.spacing,
        )
        
        tracker = DTracker(
            mask=self.mask.copy(),
            start=get_from_local_dict(self.start_point, vessel_name),
            end=get_from_local_dict(self.end_point, vessel_name),
            **func_dict, **kwargs
        )
        
        hier_path = tracker.trace_by_endpoints_v2(get_from_local_dict(self.bifur, vessel_name), kwargs.get('start_dir', None))
        _hier_path_len = np.sum([np.abs(np.array(hier_path[0][0][i+1]) - np.array(hier_path[0][0][i])) for i in range(len(hier_path[0][0])-1)], axis=0).dot(np.array(self.spacing))
        _cutoff_path_len = np.random.randint(*kwargs.get('cutoff', (_hier_path_len, _hier_path_len+1)))
        hier_path[0][0] = hier_path[0][0][:round(len(hier_path[0][0]) * _cutoff_path_len / _hier_path_len)]
        
        for k in range(max(hier_path.keys()), -1, -1):
            if k in hier_path.keys():
                hier_path[k + vessel_level] = hier_path.pop(k)
        
        art_intensity_mean = np.mean(self.ct[self.mask == LABEL_MAPPING['aorta']])
        art_intensity_std = np.std(self.ct[self.mask == LABEL_MAPPING['aorta']])
        vein_intensity_mean = np.mean(self.ct[self.mask == LABEL_MAPPING['inferior_vena_cava']])
        vein_intensity_std = np.std(self.ct[self.mask == LABEL_MAPPING['inferior_vena_cava']])
        
        postprocessor = PipelinePostprocessor(
            hier_path, self.mask, self.blank, self.spacing,
            hu_mean=(art_intensity_mean, vein_intensity_mean),
            hu_std=(art_intensity_std, vein_intensity_std)
        )
        postprocessor.postprocess_()
        foreground = postprocessor.foreground
        background = postprocessor.background
        self.foreground[(foreground > 0) & (self.mask == 0)] = foreground[(foreground > 0) & (self.mask == 0)]
        self.mask[(foreground > 0) & (self.mask == 0)] = 255
        self.blank = background
        self._register(vessel_name[0], postprocessor._vessel)
        return postprocessor._vessel


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, choices=["cmu", "totalseg"])
    parser.add_argument("--save_root", type=str, default="Dataset000CMU")
    args = parser.parse_args()
    desired_procs = 16
    # sometimes needs to be <= 16 to prevent soft lockup and slow responding
    # a soft lockup is probably because of:
    # 1) high disk IO / CPU utilization https://forums.centos.org/viewtopic.php?t=60087
    # ...) many other possible reasons, but most probably 1)
    
    if args.dataset == "cmu":
        # cmu dataset
        assert 'cmu' in args.save_root.lower()
        cmu_processed = "/data/dataset/cmu/v2"
        zipped_dirs = [_ for _ in sorted(listdir(join(cmu_processed, "image")))]
        makedir_or_dirs(join(os.environ.get("nnUNet_raw"),
                         args.save_root,
                         "imagesTr"), destory_on_exist=False)
        makedir_or_dirs(join(os.environ.get("nnUNet_raw"),
                         args.save_root,
                         "labelsTr"), destory_on_exist=False)
        
        shutil.copytree('/'.join(abspath(__file__).split('/')[:-2]), 
                        join(os.environ.get("nnUNet_raw"), args.save_root, f"code_{time.strftime('%Y-%m-%d_%H-%M-%S')}"), 
                        ignore=shutil.ignore_patterns('.git', '__pycache__'),
                        dirs_exist_ok=True)
        
        n_proc = min(desired_procs, min(mp.cpu_count(), len(zipped_dirs)))
        printf(f"[INFO] expecting a total of {n_proc} worker processes")
        process_pool = []
        for _ in range(n_proc):
            proc = mp.Process(target=augment_cmu, args=(cycle(zipped_dirs[_::n_proc]), _))
            proc.start()
            time.sleep(30)
            process_pool.append(proc)
        for proc in process_pool:
            proc.join()
    
    elif args.dataset == "totalseg":
        # totalseg dataset
        assert 'totalseg' in args.save_root.lower()
        totalseg_processed = "/data/dataset/vessel_augmentation"
        zipped_dirs = [_ for _ in sorted(listdir(join(totalseg_processed, "image")))]
        makedir_or_dirs(join(os.environ.get("nnUNet_raw"),
                         args.save_root,
                         "imagesTr"), destory_on_exist=False)
        makedir_or_dirs(join(os.environ.get("nnUNet_raw"),
                         args.save_root,
                         "labelsTr"), destory_on_exist=False)
        shutil.copytree('/'.join(abspath(__file__).split('/')[:-2]),
                        join(os.environ.get("nnUNet_raw"), args.save_root, f"code_{time.strftime('%Y-%m-%d_%H-%M-%S')}"), 
                        ignore=shutil.ignore_patterns('.git', '__pycache__'),
                        dirs_exist_ok=True)
        
        process_pool = []
        n_proc = min(desired_procs, min(mp.cpu_count(), len(zipped_dirs)))
        printf(f"[INFO] expecting a total of {n_proc} worker processes")
        for _ in range(n_proc):
            proc = mp.Process(target=augment_totalseg, args=(zipped_dirs[_::n_proc], _))
            proc.start()
            time.sleep(30)
            process_pool.append(proc)
        for proc in process_pool:
            proc.join()
    else:
        raise NotImplementedError(f"[ERROR] dataset {args.dataset} is not valid")
    