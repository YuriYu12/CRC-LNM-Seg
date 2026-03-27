import numpy as np

from os.path import *
from VesselGen.trace.A_star import astar
from scipy.interpolate import splprep, splev
from scipy import ndimage
from VesselGen.funcbase import Box, printf


def vessel_label_mapping(aug_label):
    if aug_label % 10 == 9: return aug_label // 20 + 1  # sup & inf_mesenteric_artery_vein_(0) & 1 (1, 2)
    elif aug_label % 10 <= 1: return aug_label // 20 + 3   # sup & inf_mesenteric_artery_vein_2 (3, 4)
    elif aug_label % 10 > 1: return aug_label // 20 + 5   # sup & inf_mesenteric_artery_vein_3 (5, 6)
    

def euclidean_distance(seed, dis_threshold, spacing=[1, 1, 1]):
    threshold = dis_threshold
    if(seed.sum() > 0):
        euc_dis = ndimage.distance_transform_edt(seed==0, sampling=spacing)
        euc_dis[euc_dis > threshold] = threshold
        dis = 1-euc_dis/threshold
    else:
        dis = np.zeros_like(seed, np.float32)
    return dis


class PipelinePostprocessor:
    def __init__(self, path, mask, blank, spacing=None, **kwargs):
        
        self.mask = mask
        self.blank = blank
        self._vessel = path
        self.spacing = spacing
        self._astar = astar(**kwargs)
        self.pois = np.zeros(self.mask.shape, dtype=np.uint8)
        hu_means, hu_stds = kwargs.get('hu_mean', [80, 80]), kwargs.get('hu_std', [80, 80])
        
        self.art_hu = (hu_means[0], hu_stds[0])
        self.vein_hu = (hu_means[1], hu_stds[1])
        self.box = Box(self.mask > 0, outline=2)
        # self.mask[self.mask == 255] = 0
        
    def _bspline(self, path, n_samples=50, degree=3):
        path = np.array(self._fill_path(path))
        # path = np.array(path)
        if len(path) <= n_samples:
            n_samples = max(min(len(path) // 2, len(path) - 1), 3)
        sampled_knots = np.linspace(0, 1, n_samples)
        sampled_knots = np.array([path[round(_ * (len(path) - 1))] for _ in sampled_knots]).T
        tck, _ = splprep(path.T, k=degree, t=sampled_knots)
        curve_points = np.linspace(0, 1, len(path))
        curve_points = np.round(splev(curve_points, tck)).T.astype(int)
        return self._fill_path(curve_points.tolist())
            
    def _fill_path(self, path: list):
        filled_path = []
        for ip in range(len(path) - 1):
            s_ = path[ip]
            e_ = path[ip + 1]
            if self.mask[*s_] != 0 and False:  # ip > 10 / np.mean(self.spacing):
                continue
            if len(filled_path) > 0:
                filled_path.pop(-1)
            filled_path.extend(self._astar(self.mask > 0, s_, e_))
                    
        return filled_path
            
    def postprocess_(self, r0=1.5, r1=1.3, r2=1.1, r3=.8, outline=10):
        
        self.vessel = np.zeros(self.mask.shape)
        for _bifur in self._vessel.keys():
            paths = self._vessel[_bifur]
            for _ip, _path in enumerate(paths):
                try:
                    _path = self._bspline(_path, degree=3 if _bifur % 10 < 5 else 1)
                except Exception as e:
                    printf(f"[WARNING] bspline error: {e}")
                    _path = self._fill_path(_path)
                for ip, p in enumerate(_path):
                    self.vessel[*p] = _bifur + 1
        
        foreground = self.vessel.copy()
        background = self.blank.copy()
        
        def vessel_radius_mapping(aug_label):
            nonlocal r0, r1, r2, r3
            if aug_label == 0: return lambda x: r0 + x * (r1 - r0)
            elif aug_label % 10 == 9: return lambda x: 2 * (r0 + x * (r1 - r0))
            elif aug_label % 10 == 1: return lambda x: r1 + x * (r2 - r1)
            elif aug_label % 10 == 2: return lambda x: r2 + x * (r3 - r2)
            else: return lambda x: r3
            
        for _bifur in reversed(self._vessel.keys()):
            paths = self._vessel[_bifur]
            func = vessel_radius_mapping(_bifur)
                
            for _path in paths:
                for index in range(len(_path)):
                    x, y, z = _path[index]
                    lil_bbox = self.box.get_lil_box(self.vessel, (x, y, z), outline, return_slice=True)
                    # r = func((index + sum(anchor_len[:(_bifur % 10) + 1])) / sum(anchor_len)) if _bifur % 10 < 5 else 3 * r0 - 2 * r0 * index / len(_path)
                    r = func(index / len(_path))
                    
                    cl_crop = self.vessel[*lil_bbox]  # centerline
                    fg_crop = foreground[*lil_bbox]  # foreground
                    bg_crop = background[*lil_bbox]  # background
                    msk_crop = self.mask[*lil_bbox]  # masked organs (>0) and vessels (255)
                    
                    cl_map = 1 - (cl_crop == _bifur + 1) * 1.
                    cl_dst = ndimage.distance_transform_edt(cl_map, self.spacing)
                    # 1) maybe add randomness since vessel can't be perfect cylinder
                    r_ = (np.random.random() + .5) * r  # / np.mean(self.spacing)
                    # 2) avoid organ collision
                    organ_map = 1 - (msk_crop > 0) * 1.
                    organ_dst = ndimage.distance_transform_edt(organ_map, self.spacing)
                    
                    dst = (cl_dst < r_) & (msk_crop == 0)
                    fg_crop[dst] = vessel_label_mapping(_bifur)
                    if _bifur % 20 > 10:
                        bg_crop[dst] = (np.random.random((dst.sum(),)) - 0.5) * self.vein_hu[1] + self.vein_hu[0] * 0.8
                        # bg_crop[blur_region] = bg_crop[blur_region] * 0.3 + np.random.normal(blur_anchor.mean(), blur_anchor.std() / 2, blur_region.sum()) * 0.8
                    else:
                        bg_crop[dst] = (np.random.random((dst.sum(),)) - 0.5) * self.art_hu[1] + self.art_hu[0] * 0.8
                        # bg_crop[blur_region] = bg_crop[blur_region] * 0.3 + np.random.normal(blur_anchor.mean(), blur_anchor.std() / 2, blur_region.sum()) * 0.8
        
                    foreground[*lil_bbox] = fg_crop
                    background[*lil_bbox] = bg_crop
        
        self.foreground = foreground
        self.background = background
        self._process_hu_anchors()
    
    def add_poi(self, label_map_or_indices, start_label):
        label_map_or_indices = np.array(label_map_or_indices)
        if label_map_or_indices.ndim == 1:
            self.pois[*label_map_or_indices] = start_label
        elif label_map_or_indices.ndim == 2:
            self.pois[*label_map_or_indices.T.tolist()] = start_label
        elif label_map_or_indices.ndim == 3:
            for i_label, label in enumerate(np.unique(label_map_or_indices)):
                if label != 0:
                    self.pois[label_map_or_indices == label] = i_label + start_label
    
    def upsample(self, upsample_spacing=(1, .6, .6)):
        
        resize_coeff = np.array(self.spacing) / np.array(upsample_spacing)
        self.mask = ndimage.zoom(self.mask, resize_coeff, order=0)
        self.pois = ndimage.zoom(self.pois, resize_coeff, order=0)
        self.blank = ndimage.zoom(self.blank, resize_coeff, order=3)
        # self.vessel = zoom(self.vessel, resize_coeff, order=0)
        
        for _bifur in self._vessel.keys():
            paths = self._vessel[_bifur]
            for _ip, _path in enumerate(paths):
                for ip, p in enumerate(_path):
                    transform_p = tuple(np.round(np.array(p) * resize_coeff).astype(int))
                    _path[ip] = transform_p
                    
        self.spacing = upsample_spacing
        
    def _process_hu_anchors(self):
        assert self.foreground.shape == self.mask.shape == self.background.shape
        
        artery_hu_anchor = self.art_hu
        vein_hu_anchor = self.vein_hu
        render_vessel_fn = lambda param, size: (np.random.rand(size) - 0.5) * param[1] + param[0]
        
        raw_foreground = self.foreground
        foreground = raw_foreground.copy()
        background = self.background
        
        mask = self.mask

        artery_hu_anchor = [background[mask == 7].mean() * 0.8,
                                    background[mask == 7].std()]
        vein_hu_anchor = [background[mask == 8].mean() * 0.8,
                            background[mask == 8].std()]
        render_vessel_fn = lambda param, size: (np.random.random(size) - 0.5) * param[1] + param[0]

        def salt_and_pepper(salt_size, pepper_size, typename="artery"):
            if typename == "artery": foreground_hu_anchor = artery_hu_anchor
            elif typename == "vein": foreground_hu_anchor = vein_hu_anchor
            
            salt = render_vessel_fn((foreground_hu_anchor[0] + foreground_hu_anchor[1] - 10, 20), salt_size)
            pepper = render_vessel_fn((foreground_hu_anchor[0] - foreground_hu_anchor[1] - 10, 20), pepper_size)
            noise = np.concatenate([salt, pepper])
            np.random.shuffle(noise)
            return noise

        bounding_box = Box(foreground, outline=5)
        cp_foreground = self.foreground[*bounding_box.cropped_slice]
        combine = background[*bounding_box.cropped_slice].copy()
        
        artery_foreground = (cp_foreground > 0) & (cp_foreground % 20 < 10)
        vein_foreground = cp_foreground % 20 >= 10
        artery_size = artery_foreground.sum()
        vein_size = vein_foreground.sum()
        combine[artery_foreground] = salt_and_pepper(salt_size=(round(artery_size * 0.7),),
                                                    pepper_size=(artery_size-round(artery_size * 0.7)),
                                                    typename="artery")
        combine[vein_foreground] = salt_and_pepper(salt_size=(round(vein_size * 0.7),),
                                                    pepper_size=(vein_size-round(vein_size * 0.7)),
                                                    typename="vein")
        
        combine = ndimage.zoom(combine, 2., order=0)
        cp_foreground_ = ndimage.zoom(cp_foreground, 2, order=0)
        smooth = ndimage.gaussian_filter(combine, sigma=1.2, radius=3)
        combine[cp_foreground_ > 0] = smooth[cp_foreground_ > 0]
        combine = ndimage.zoom(combine, .5, order=3)

        background[*bounding_box.cropped_slice][cp_foreground] = combine[cp_foreground]

        dist = ndimage.distance_transform_edt(foreground > 0, sampling=self.spacing)
        surface = (dist > 0) & (dist < .8)
        background[surface] = background[surface] * 0.7
        
        foreground[*bounding_box.cropped_slice][:] = cp_foreground
        erosion_vessel_mask_array = ndimage.binary_erosion(foreground, structure=np.ones((2,2,2)))
        dis = euclidean_distance(erosion_vessel_mask_array, 3, spacing=self.spacing)
        background = background * dis + self.background * (1 - dis)
        
        new_foreground = np.zeros(foreground.shape, dtype=np.uint8)
        for i in range(1, 7):
            new_foreground[(raw_foreground == i) & (foreground > 0)] = i
            
        self.foreground = new_foreground
        self.background = background