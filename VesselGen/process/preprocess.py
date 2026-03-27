import os
import sys
sys.path.append(os.getcwd())

import time
import json, ujson
import shutil
import nibabel
import pathlib
import numpy as np
import SimpleITK as sitk
import multiprocessing as mp

from os.path import *
from tqdm import tqdm
from VesselGen.macros import LABEL_MAPPING
from VesselGen.process.segment_colon_sections import process_colon
from VesselGen.funcbase import save_nifti, Box, slice_orientation, \
    find_largest_connected_components, window_norm
from scipy.ndimage import binary_fill_holes, generate_binary_structure, \
    label, binary_dilation, binary_erosion, gaussian_filter, distance_transform_edt
from scipy.interpolate import splprep, splev
from VesselGen.trace.frangi import frangi

args = None


def makedir_or_dirs(path, destory_on_exist=False):
    if exists(path):
        if destory_on_exist:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def listdir(path):
    return [join(path, subpath) for subpath in os.listdir(path) if not subpath.startswith('.')]


def cleanup(*foldernames):
    id_foldernames = [[int(fname.split('/')[-1].split('_')[1]) for fname in foldername] for foldername in foldernames]
    all_foldernames = [sorted(foldername) for foldername in id_foldernames]
    unique_foldernames = [[] for _ in range(len(id_foldernames))]
    indices = [0] * len(id_foldernames)

    while True:
        assertion = max([indices[i_foldername] - len(all_foldernames[i_foldername]) for i_foldername in range(len(all_foldernames))])
        if assertion >= 0: break
        compr = [foldername[indices[i_foldername]] for i_foldername, foldername in enumerate(all_foldernames)]
        for i_foldername, foldername in enumerate(all_foldernames):
            index = indices[i_foldername]
            if foldername[index] * len(indices) < sum(compr):
                indices[i_foldername] += 1
                unique_foldernames[i_foldername].append(foldername[index])
            elif foldername[index] * len(indices) == sum(compr):
                indices[i_foldername] += 1
    
    for i_foldername, foldername in enumerate(all_foldernames):
        index = indices[i_foldername]
        if index < len(foldername):
            unique_foldernames[i_foldername].extend(foldername[index:])
            
    for i_foldername, foldername in enumerate(unique_foldernames):
        prefix = "/".join(foldernames[i_foldername][0].split('/')[:-1] + foldernames[i_foldername][0].split('/')[-1].split('_')[:1])
        suffix = ("_0000" if len(foldernames[i_foldername][0].split('/')[-1].split('_')) > 2 else "") + ".nii.gz"
        for index in foldername:
            fname = prefix + f"_{index:05d}" + suffix
            os.remove(fname)
            print(f"removed {fname}")


def get_background(msk, ct, conn_ths=500):
    
    msk = msk.copy()
    res = np.zeros(ct.shape, dtype=np.uint8)
    main_vessel_conn_tree = (msk == LABEL_MAPPING['aorta']) | ((msk == LABEL_MAPPING['inferior_vena_cava']))
    abdomen_bbox = Box(msk == LABEL_MAPPING['colon'], 1)
    if abdomen_bbox.cropped_slice is None:
        abdomen_bbox = Box((msk >= LABEL_MAPPING['ascendant_colon']) & (msk <= LABEL_MAPPING['rectum']), 1)
    candidate = np.zeros(ct.shape, dtype=int)

    abdomen_interior_2 = abdomen_bbox.cropped_slice
    candidate_bbox = candidate[*abdomen_interior_2]
    candidate_bbox[main_vessel_conn_tree[*abdomen_interior_2]] = True
    aorta_mean = np.argwhere(main_vessel_conn_tree[*abdomen_interior_2]).mean(axis=0)[1]
    
    ct_hu_bbox_cap = ct[*abdomen_interior_2] > 70
    ct_organ_elim = ct_hu_bbox_cap & (msk[*abdomen_interior_2] == 0)
    """ct_o, m = label(ct_organ_elim)
    ct_o_counts = np.bincount(ct_o.flatten())
    seeds_o = [i for i in tqdm(range(1, m)) \
        if 10 < ct_o_counts[i] < 1e5 and np.argwhere(ct_o == i).mean(axis=0)[1] > aorta_mean]"""
    
    itr = 1
    candidate_record1 = None
    candidate_record2 = None
    n_ori_candidate_bbox = candidate_bbox.sum()
    n_ct_organ_elim = np.bincount(label(ct_organ_elim)[0].flatten())[1:].max()
    while itr < 1e5:
        candidate_bbox = binary_dilation(candidate_bbox, mask=ct_organ_elim)
        if candidate_bbox.sum() - n_ori_candidate_bbox >= min(0.4 * n_ct_organ_elim, 1e5) and candidate_record1 is None:
            candidate_record1 = [itr, candidate_bbox]
        if candidate_bbox.sum() - n_ori_candidate_bbox >= min(0.6 * n_ct_organ_elim, 3e5): 
            candidate_record2 = [itr, candidate_bbox]
            break
        itr += 1
        print(f"{itr} / 1e5", end="\r")
    ct_v, n = label(candidate_record2[1] & ~candidate_record1[1])
    
    ct_v_counts = np.bincount(ct_v.flatten())
    seeds_v = [i for i in tqdm(range(1, n)) \
        if 1 < ct_v_counts[i] < n * (candidate_record2[0] - candidate_record1[0]) and \
            np.argwhere(ct_v == i).mean(axis=0)[1] > aorta_mean]
    seeds_v.extend([i for i in tqdm(range(1, n)) if 1 == (ct_v == i).sum() and np.argwhere(ct_v == i)[0][1] > aorta_mean])
    print(f"{len(seeds_v)} seeds")
    
    msk[distance_transform_edt(msk == 0) < 3] = 1
    dst_to_ct_hu_bbox_cap_1 = distance_transform_edt(~ct_hu_bbox_cap) < 10
    ct_hu_bbox_cap_2 = (ct[*abdomen_interior_2] > 0) & dst_to_ct_hu_bbox_cap_1
    ct_organ_elim = ct_hu_bbox_cap_2 & (msk[*abdomen_interior_2] == 0)
    
    # find small clusters that are in front of aorta in y axis
    itr = 1
    candidate_bbox[...] = 0
    """for index, i in enumerate(seeds_o): 
        candidate_bbox[*np.argwhere(ct_o == i)[np.random.randint(0, (ct_o == i).sum())]] = index"""
    max_index = candidate_bbox.max() + 1
    for index, i in enumerate(seeds_v): 
        candidate_bbox[ct_v == i] = index + max_index
        
    candidate_bbox_dil = candidate_bbox.copy()
    n_ori_candidate_bbox = 0
    max_preserve_pt = 3e5
    while itr < 1e5:
        candidate_bbox_dil = binary_dilation(candidate_bbox_dil, mask=ct_organ_elim)
        if itr % 10 == 0:
            flag = True
            fronts, n = label(candidate_bbox_dil & ~candidate_bbox)
            n_pt_fronts = np.unique(fronts, return_counts=True)
            for i_front, (front, front_pts) in enumerate(zip(*n_pt_fronts)):
                if front_pts < 3000:
                    flag = False
                    candidate_bbox[fronts == i_front] = candidate_bbox_dil[fronts == front]
            candidate_bbox_dil = candidate_bbox.copy()
            if flag: 
                break
            
        if n_ori_candidate_bbox == candidate_bbox_dil.sum(): 
            break
        n_ori_candidate_bbox = candidate_bbox_dil.sum()
        if candidate_bbox.sum() > max_preserve_pt: 
            break
        itr += 1
        print(f"{itr} / 1e5; {candidate_bbox.sum()} / {max_preserve_pt}", end="\r")
    candidate[*abdomen_interior_2] = candidate_bbox
    print()
    a_v, n = label(candidate)
    a_v_counts = np.bincount(a_v.flatten())[1:]
    
    ct_v_hu_conn_cut = ct.copy()
    for i, ni in enumerate(a_v_counts):
        if ni > conn_ths:
            res[a_v == i + 1] = 1
    
    res = binary_dilation(res).astype(np.uint8)
    ct_v_hu_conn_cut[res == 1] = np.random.random((res.sum(),)) * 60 - 130
    ct_v_hu_conn_cut[msk > 0] = ct[msk > 0]
    res_surface = res & (distance_transform_edt(res == 1) < 1.5)
    bbox = Box(res == 1, outline=10)
    for pt in np.argwhere(res_surface == 1):
        slices = bbox.get_lil_box(ct_v_hu_conn_cut, pt, 1, return_slice=True)
        ct_slice = ct_v_hu_conn_cut[*slices]
        res_slice = res[*slices]
        ct_slice[res_slice == 1] = gaussian_filter(ct_slice, sigma=.5)[res_slice == 1]
        continue
    # ct_v_hu_conn_cut[gaussian_surrounding] = gaussian_filter(ct_v_hu_conn_cut[gaussian_surrounding], sigma=10)
        
    return ct_v_hu_conn_cut, res


def get_background_v2(mask, ct, size_mm=20, remove_prop=.2, hu_thresh=70):
    """divide the abdomen volume into multiple 2x2x2 cm cubes, for cube that
        1) contains hu>-50 structures, and 
        2) the largest connected domain within the cube <30% of total volume
        label all the hu>-50 structure in the cube as vessel candidates, 
    end for
    find largest few connected domains that span the largest as true vessel

    Args:
        msk (np.ndarray): totalseg segmented, `maybe` processed (colon, spine) ct mask
        ct (np.ndarray): original ct
        size_mm (int or float): irregularized side length for divided cube
        remove_prop (float): <1, if conn domain > this threshold, then discard the conn domain
        hu_thresh (int or float): hu threshold for vessel
    """
    z_sacrum_max = np.argwhere(mask == LABEL_MAPPING["sacrum"])[:, 0].max()
    spine_invalid_patches = np.asarray([_ for _ in np.argwhere(mask == LABEL_MAPPING["patched_spine"]) if _[0] < z_sacrum_max])
    mask[*spine_invalid_patches.T] = 0
    
    background = ct.copy()
    abdominal_bbox = Box((mask >= LABEL_MAPPING['ascendant_colon']) & (mask <= LABEL_MAPPING['descendant_colon']), 1)
    if abdominal_bbox.cropped_slice is None:
        abdominal_bbox = Box(mask == LABEL_MAPPING['colon'], 1)
    abdomen_ct = background[*abdominal_bbox.cropped_slice]
    abdomen_mask = mask[*abdominal_bbox.cropped_slice]
    vessel_candidates = np.zeros(abdomen_ct.shape, dtype=np.uint8)
    for z in tqdm(range(size_mm // 2, abdomen_ct.shape[0], size_mm)):
        for y in range(size_mm // 2, abdomen_ct.shape[1], size_mm):
            for x in range(size_mm // 2, abdomen_ct.shape[2], size_mm):
                slice_ = [slice(z - size_mm // 2, z + size_mm // 2, None),
                          slice(y - size_mm // 2, y + size_mm // 2, None),
                          slice(x - size_mm // 2, x + size_mm // 2, None)]
                focus_ct_box = abdomen_ct[*slice_]
                focus_mask_box = abdomen_mask[*slice_]
                focus_box_size = np.prod(focus_ct_box.shape)
                focus_box_candidates = (focus_ct_box > hu_thresh) & (focus_mask_box == 0)
                
                primary_structure = find_largest_connected_components(focus_box_candidates, n=1)
                while (primary_structure > 0).sum() > remove_prop * focus_box_size:
                    focus_box_candidates[primary_structure == 1] = False
                    primary_structure = find_largest_connected_components(focus_box_candidates, n=1)
                vessel_candidates[*slice_][focus_box_candidates] = 1
    
    # 2) the largest connected areas are vessels
    vessel_proposal = np.zeros(ct.shape, dtype=np.uint8)
    vessel_proposal[*abdominal_bbox.cropped_slice] = find_largest_connected_components(vessel_candidates, n=1).astype(np.uint8)
    abdomen_vessel_prop = vessel_proposal[*abdominal_bbox.cropped_slice]
    
    # 3) gaussian blurring of proposed vessel region
    abdomen_vessel_prop = binary_dilation(abdomen_vessel_prop > 0, iterations=3).astype(np.uint8)
    abdomen_ct[abdomen_vessel_prop > 0] = np.random.random(((abdomen_vessel_prop > 0).sum(),)) * 60 - 130
    vessel_surface = abdomen_vessel_prop & (distance_transform_edt(abdomen_vessel_prop) < 1.5)
    for pt in tqdm(np.argwhere(vessel_surface)):
        slice_ = abdominal_bbox.get_lil_box(abdomen_ct, pt, 1, return_slice=True)
        ct_slice = abdomen_ct[*slice_]
        vessel_slice = abdomen_vessel_prop[*slice_]
        ct_slice[vessel_slice > 0] = gaussian_filter(ct_slice, sigma=.5)[vessel_slice > 0]
        continue
    # vessel_proposal[mask == 111] = 255
    return background, vessel_proposal


def preprocess(_names):
    
    for name in _names:
        try:
            print(f"worker {os.getpid()} is processing file {name['fname']}")
            fname = name["fname"]
            save_dir = name["save_dir"]
            anchor = sitk.ReadImage(name["ct_path"])
            ct = sitk.GetArrayFromImage(anchor)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(name["mask_path"]))
            scribble = None if name["scribble_path"] is None \
                else sitk.GetArrayFromImage(sitk.ReadImage(name["scribble_path"]))[:, ::-1]
            section_mask = None if name["section_mask_path"] is None \
                else sitk.GetArrayFromImage(sitk.ReadImage(name["section_mask_path"]))
            assert scribble is not None or section_mask is not None
            
            if not exists(join(save_dir, 'image', fname)):
                save_nifti(ct, join(save_dir, 'image', fname), anchor)
            
            if not exists(join(save_dir, 'mask', fname)) or True:
                colon_mask = (mask == LABEL_MAPPING['colon']).astype(np.uint8)
                if section_mask is None:
                    section_mask, m = process_colon(scribble, colon_mask, [104 + _ for _ in range(6)], 'pca')
                    if m != 'pca': raise RuntimeWarning("pca algo not available")
                mask[colon_mask > 0] = section_mask[colon_mask > 0]
                conn_spine(mask)
                save_nifti(mask, join(save_dir, 'mask', fname), anchor)
            else:
                mask = sitk.GetArrayFromImage(sitk.ReadImage(join(save_dir, 'mask', fname)))
            
            """upper_limit = np.asarray(np.argwhere(mask == LABEL_MAPPING['aorta']))
            upper_limit = upper_limit[np.where(upper_limit[:, 0] == np.min(np.argwhere(mask == LABEL_MAPPING['portal_vein_and_splenic_vein'])[:, 0]))]
            upper_x, upper_y, upper_z = np.mean(upper_limit, axis=0, dtype=int)
            # [z-, z+ (in itksnap LRAPIL: I: S), y-, y+ (P: A), x-, x+ (L: R)] in LPI format
            ct_cutoff = ct[max(0, upper_x - Z(100)): min(ct.shape[0] - 1, upper_x + Z(50)),
                        max(0, upper_y - Y(30)): min(ct.shape[1] - 1, upper_y + Y(80)),
                        max(0, upper_z - X(30)): min(ct.shape[2] - 1, upper_z + X(70))]"""
            
            if not exists(join(save_dir, 'blank', fname)) or True:
                blank_v2, vis = get_background_v2(mask, ct)
                save_nifti(blank_v2, join(save_dir, 'blank', fname), anchor)
                # save_nifti(blank_v2, join(save_dir, 'trash', f'blank_v2.nii.gz'), anchor)
                # save_nifti(vis, join(save_dir, 'trash', f'blank_v2_f.nii.gz'), anchor)
        except Exception as e:
            print(e)
            continue
    
    # legacy: ori ct pos <space> cut ct pos
    """cut_x = Z(100, round) if Z(100, round) <= upper_x else upper_x
    cut_y = Y(30, round) if Y(30, round) <= upper_y else upper_y
    cut_z = X(30, round) if X(30, round) <= upper_z else upper_z
    anchor_point = f"{upper_x}_{upper_y}_{upper_z} {cut_x}_{cut_y}_{cut_z}"
    save_nifti(ct, join(save_dir, 'seginputs', f'{fname}_{anchor_point}.nii.gz'), anchor)
    save_nifti(ct, join(save_dir, 'seginputs', 'alias', nnUNet_name_fn), anchor)"""


def conn_spine(mask, spine_thresh=1000):
    totalseg_spine = np.zeros(mask.shape)
    spine_id = [f"vertebrae_T{x}" for x in range(4, 13)] + [f"vertebrae_L{x}" for x in range(1, 6)]
    for i_spine in spine_id: totalseg_spine[mask == LABEL_MAPPING[i_spine]] = 1
    totalseg_spine[slice_orientation(totalseg_spine, orient=0, direction=0, half=1, percentile=40) > 0] = 0
    totalseg_spine[mask == LABEL_MAPPING["sacrum"]] = 1
    centerline_spine = []  # approx
    radius_spine = []  # approx as an oval shape, in unit of grid points rather than mm
    radius_mapping = {}
    # 1) get slices in the SI axis
    for layer in range(totalseg_spine.shape[0]):
        sliced_spine = totalseg_spine[layer]
        if sliced_spine.sum() < spine_thresh: continue
        else: 
            labeled_slice, n = label(sliced_spine)
            cand = [np.argwhere(labeled_slice == i).mean(axis=0)[0] for i in range(1, n+1)]
            sliced_spine_mst = labeled_slice == (np.argmax(cand) + 1)
            if sliced_spine_mst.sum() < spine_thresh: continue
            centerline_spine.append([layer] + np.round(np.argwhere(sliced_spine_mst).mean(axis=0)).astype(int).tolist())
            radius_spine.append(np.sqrt((sliced_spine_mst > 0).sum() * 1.5 / np.pi))    # sampling 1: 1: 1.5 added to compensate the oval shape of vertebrae
            radius_mapping[layer] = radius_spine[-1]
    z_span = np.argwhere(np.any(totalseg_spine, axis=(1, 2)))[[0, -1]]
    # 2) use Bspine to interpolate the spine centerline
    sampled_percentiles = np.linspace(0, 1, min(50, len(radius_spine)))
    sampled_knots = np.array([centerline_spine[round(_ * (len(centerline_spine) - 1))] for _ in sampled_percentiles]).T
    sampled_weights = np.array([radius_spine[round(_ * (len(radius_spine) - 1))] for _ in sampled_percentiles])
    tck, _ = splprep(sampled_knots,
                     k=2,
                     s=100,
                     w=np.exp(2 * sampled_weights / max(radius_spine)))
    curve = np.round(splev(np.linspace(0, 1, (z_span[1] - z_span[0] + 1)[0]), tck)).T.astype(int)
    totalseg_spine[*curve.T.tolist()] = 2
    totalseg_spine[binary_dilation(totalseg_spine == 2, iterations=2) > 0] = 2
    # 3) expand centerline to the recorded radius
    for centerpoint in curve:
        sliced_spine = totalseg_spine[centerpoint[0]]
        dst = np.ones(sliced_spine.shape)
        dst[centerpoint[1], centerpoint[2]] = 0
        dst_inside = distance_transform_edt(dst, sampling=(1.5, 1))  # sampling 1: 1: 1.5 added to compensate the oval shape of vertebrae
        r = np.mean([v for k, v in radius_mapping.items() if abs(k - centerpoint[0]) < 20])
        sliced_spine[(dst_inside <= r * 1.2) & (sliced_spine == 0)] = 2
    # 4) save as label 111 in totalseg labels, can be accessed via "patched_spine"
    def rect_back(n):
        strc = np.zeros((n, n, n))
        strc[n//2:, :n//2, :] = 1
        return strc.astype(bool)
    def rect_front(n):
        strc = np.zeros((n, n, n))
        strc[:n//2, :n//2, :] = 1
        return strc.astype(bool)
    totalseg_spine[binary_dilation(totalseg_spine == 2, structure=rect_back(6), iterations=8)] = 2
    totalseg_spine[binary_dilation(totalseg_spine == 2, structure=rect_front(6), iterations=2)] = 2
    mask[(totalseg_spine == 2) & (mask == 0)] = 111
    return totalseg_spine
    

def seg(save_dir):
    os.system(f"nnUNetv2_predict -i {join(save_dir, 'image')} -o {join(save_dir, 'seg')} -c 3d_fullres -d 7 -f 0 -chk checkpoint_best.pth")
    
    
def preprocessor(_queue, n_proc=24):
    process_pool = []
    print(f"expecting a total of {n_proc} workers")
    for i_proc in range(n_proc):
        p = mp.Process(target=preprocess, args=(_queue[i_proc::n_proc],))
        p.start()
        time.sleep(20)
        process_pool.append(p)
    for i_proc in range(n_proc):
        process_pool[i_proc].join()
    
    
def preprocess_cmu():
    # cmu dataset
    BASE_PATH = "/data/dataset/cmu/v1/original"
    SCRIBBLE_PATH = "/data/dataset/cmu/v1/scribble"
    MOD_PATH = "/data/dataset/cmu/v2"
    makedir_or_dirs(join(MOD_PATH, 'seg'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'mask'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'blank'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'image'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'trash'), destory_on_exist=False)

    people_dict = []
    people = listdir(BASE_PATH)
    people.sort(key=lambda x: x.split('/')[-1].split(' ')[0])
    # people = people[:10]
    
    with open(join(MOD_PATH, "varname2path.json"), 'w') as fp:
        for iperson, person in tqdm(enumerate(people)):
            person_folder = join(BASE_PATH, person, listdir(person)[0])
            person_files = listdir(person_folder)
            phases = []
            for person_file in person_files:
                if person_file.endswith("_totalseg.nii.gz"):
                    phases.append(person_file)
            if len(phases) == 0: continue
            for iphase, phase in enumerate(phases):
                totalseg = phase
                ct = totalseg.replace("_totalseg", "")
                scribble_path = join(SCRIBBLE_PATH, "@@".join(ct.replace(BASE_PATH, '').split('/')[1:]))
                scribble_path = scribble_path.replace(".nii.gz", "scrib_llm_nois0.1_colon.nii.gz")
                
                fname = f"Dataset000CMU_{iperson:05d}{iphase}_0000.nii.gz"
                person_dict = dict(
                    fname=fname,
                    ct_path=ct,
                    mask_path=totalseg,
                    scribble_path=scribble_path,
                    section_mask_path=None, save_dir=MOD_PATH
                )

            people_dict.append(person_dict)
            json.dump(people_dict, fp)
        
    preprocessor(people_dict)
    # seg_(MOD_)
        

def preprocess_cmu_v2(people_dict=None):
    # cmu dataset
    BASE_PATH = "/data/dataset/cmu/v1/original"
    SCRIBBLE_PATH = "/data/dataset/cmu/v1/scribble"
    MOD_PATH = "/data/dataset/cmu/v2"
    makedir_or_dirs(join(MOD_PATH, 'seg'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'mask'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'blank'), destory_on_exist=True)
    makedir_or_dirs(join(MOD_PATH, 'image'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'trash'), destory_on_exist=False)
    
    if people_dict is not None:
        with open(people_dict, 'r') as fp:
            preprocessor(json.load(fp))
    else:
        people_dict = []
        people = listdir(BASE_PATH)
        people.sort(key=lambda x: x.split('/')[-1].split(' ')[0])
        
        # 1) get the images with brightest portal_and_splenic vein
        iinstance = 0
        for iperson, person in tqdm(enumerate(people), desc="choosing CE instances", total=len(people)):
            person_folder = join(BASE_PATH, person, listdir(person)[0])
            person_files = listdir(person_folder)
            phases = []
            for person_file in person_files:
                if person_file.endswith("_totalseg.nii.gz"):
                    phases.append(person_file)
            if len(phases) == 0: continue
            i_candidate_phasal_imgs = []
            max_spl_brightness = 0
            max_spl_brightness_pos = 0
            for iphase, phase in enumerate(phases):
                ct_path = phase.replace("_totalseg", "")
                ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(phase))
                spl_brightness = np.mean(ct[mask == LABEL_MAPPING["portal_vein_and_splenic_vein"]])
                art_brightness = np.mean(ct[mask == LABEL_MAPPING["aorta"]])
                vein_brightness = np.mean(ct[mask == LABEL_MAPPING["inferior_vena_cava"]])
                
                scribble_path = join(SCRIBBLE_PATH, "@@".join(ct_path.replace(BASE_PATH, '').split('/')[1:]))
                scribble_path = scribble_path.replace(".nii.gz", "scrib_llm_nois0.1_colon.nii.gz")
                i_candidate_phasal_imgs.append(dict(
                    fname=f"Dataset000CMU_{iinstance:05d}_0000.nii.gz",
                    ct_path=ct_path,
                    mask_path=phase,
                    scribble_path=scribble_path,
                    section_mask_path=None,
                    save_dir=MOD_PATH,
                    spl_brightness=str(spl_brightness),
                    art_brightness=str(art_brightness),
                    vein_brightness=str(vein_brightness),
                ))
                iinstance += 1
                if spl_brightness > max_spl_brightness:
                    max_spl_brightness = spl_brightness
                    max_spl_brightness_pos = iphase
                    
            people_dict.append(i_candidate_phasal_imgs[max_spl_brightness_pos])
            
        people_dict = [_ for _ in people_dict if float(_["art_brightness"]) > 100 and float(_["vein_brightness"]) > 100]
        with open(join(MOD_PATH, "varname2path.json"), "w") as fp:
            json.dump(people_dict, fp)
            
        # 2) do augmentation in candidate images only
        preprocessor(people_dict)
    
    cleanup(listdir(join(MOD_PATH, 'image')), join(MOD_PATH, 'mask'), join(MOD_PATH, 'blank'))
        
    
def preprocess_totalseg():
    # totalseg dataset (deprecated, cannot use `preprocessor()`, 
    # and `preprocess_()` has adapted to multiprocessing, 
    # so totalseg currently cannot be processed)
    from VesselGen.macros import BASE_PATH, MOD_PATH
    makedir_or_dirs(join(MOD_PATH, 'seg'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'mask'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'blank'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'image'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'trash'), destory_on_exist=False)
    
    """with open(join(BASE_DIR, 'valid_cases.txt'), 'r') as fp:
        foldernames = [_.strip() for _ in fp.readlines()]"""
    people_dict = []
    dummyfoldernames = ['s0010']
    
    for ifolder, foldername in tqdm(enumerate(dummyfoldernames)):
        if ifolder > 10: continue
        foldername = join(BASE_PATH, foldername)
        try:
            ct = sitk.GetArrayFromImage(sitk.ReadImage(join(foldername, 'ct.nii.gz')))
        except RuntimeError:
            ct = nibabel.load(join(foldername, 'ct.nii.gz')).dataobj[:].astype(np.float32).transpose(2, 1, 0)  # translate from orthogonal to LPI (zyx) format
        anchor = sitk.ReadImage(join(foldername, 'segmentations', 'colon.nii.gz'))
        colon_section_mask = sitk.GetArrayFromImage(sitk.ReadImage(join(foldername, 'colon_sections_pca.nii.gz')))
        colon_section_mask[colon_section_mask > 0] += 1
        all_masks = np.zeros(ct.shape, dtype=np.uint8)
        for sg in os.listdir(join(foldername, 'segmentations')):
            maskname = join(foldername, 'segmentations', sg)
            sgname = sg.replace('.nii.gz', '')
            organ_mask = nibabel.load(maskname).dataobj[:].transpose(2, 1, 0)
            all_masks[all_masks == 0] += organ_mask[all_masks == 0] * LABEL_MAPPING[sgname]
        
        fname = f"Dataset001Totalseg_{int(foldername.split('/')[-1].lstrip('s')):05d}_0000.nii.gz"
        preprocess(ct, all_masks, colon_section_mask, anchor, fname, MOD_PATH, section_mask=colon_section_mask)
        
    seg(MOD_PATH)


def move(people_list):
    MOD_PATH = pathlib.Path("/data/dataset/cmu/split2")
    for person_file in people_list:
        fname = person_file["fname"]
        print(f"worker {os.getpid()} is moving file {fname}")
        try:
            ct_path = pathlib.Path(person_file["ct_path"])
            mask_path = pathlib.Path(person_file["mask_path"])
            shutil.copyfile(ct_path, MOD_PATH / "image" / fname)
            
            sc_anchor = sitk.ReadImage(person_file["scribble_path"])
            scribble = sitk.GetArrayFromImage(sc_anchor)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        
            arr, _ = process_colon(scribble, (mask == 57).astype(np.uint8))
            mask[arr > 0] = arr[arr > 0]
            save_nifti(mask, MOD_PATH / "mask" / fname, sc_anchor)
        except Exception as e:
            print(f"{fname} raised error {e} at process {os.getpid()}, skipping this file")
            continue
    
    
def mover(_queue, n_proc=16):
    process_pool = []
    print(f"expecting a total of {n_proc} workers")
    for i_proc in range(n_proc):
        p = mp.Process(target=move, args=(_queue[i_proc::n_proc],))
        p.start()
        time.sleep(20)
        process_pool.append(p)
    for i_proc in range(n_proc):
        process_pool[i_proc].join()
        
        
def preprocess_cmu_split2(people_dict=None):
    # cmu dataset
    BASE_PATH = "/data/dataset/cmu/v1/original"
    SCRIBBLE_PATH = "/data/dataset/cmu/v1/scribble"
    MOD_PATH = "/data/dataset/cmu/split2"
    makedir_or_dirs(join(MOD_PATH, 'mask'), destory_on_exist=False)
    makedir_or_dirs(join(MOD_PATH, 'image'), destory_on_exist=False)
    
    if people_dict is not None and exists(people_dict):
        with open(people_dict, 'r') as fp:
            mover(json.load(fp))
    else:
        fp = open(join(MOD_PATH, "varname2path.json"), "w")
        people_dict = []
        people = listdir(BASE_PATH)
        people.sort(key=lambda x: x.split('/')[-1].split(' ')[0])
        
        # 1) get the images with brightest portal_and_splenic vein
        iinstance = 0
        for iperson, person in tqdm(enumerate(people), desc="choosing CE instances", total=len(people)):
            person_folder = join(BASE_PATH, person, listdir(person)[0])
            person_files = listdir(person_folder)
            phases = []
            for person_file in person_files:
                if person_file.endswith("_totalseg.nii.gz"):
                    phases.append(person_file)
            if len(phases) == 0: continue
            i_candidate_phasal_imgs = []
            max_spl_brightness = 0
            max_spl_brightness_pos = 0
            for iphase, phase in enumerate(phases):
                ct_path = phase.replace("_totalseg", "")
                ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
                mask = sitk.GetArrayFromImage(sitk.ReadImage(phase))
                spl_brightness = np.mean(ct[mask == LABEL_MAPPING["portal_vein_and_splenic_vein"]])
                art_brightness = np.mean(ct[mask == LABEL_MAPPING["aorta"]])
                vein_brightness = np.mean(ct[mask == LABEL_MAPPING["inferior_vena_cava"]])
                
                scribble_path = join(SCRIBBLE_PATH, "@@".join(ct_path.replace(BASE_PATH, '').split('/')[1:]))
                scribble_path = scribble_path.replace(".nii.gz", "scrib_llm_nois0.1_colon.nii.gz")
                i_candidate_phasal_imgs.append(dict(
                    fname=f"Dataset000CMU_{iinstance:05d}_0000.nii.gz",
                    ct_path=ct_path,
                    mask_path=phase,
                    scribble_path=scribble_path,
                    section_mask_path=None,
                    save_dir=MOD_PATH,
                    spl_brightness=str(spl_brightness),
                    art_brightness=str(art_brightness),
                    vein_brightness=str(vein_brightness),
                ))
                iinstance += 1
                if spl_brightness > max_spl_brightness:
                    max_spl_brightness = spl_brightness
                    max_spl_brightness_pos = iphase
                    
            # people_dict.append(i_candidate_phasal_imgs[max_spl_brightness_pos])
            fp.write(ujson.dumps(i_candidate_phasal_imgs[max_spl_brightness_pos]) + "\n")
            
        mover(people_dict)

    cleanup(listdir(join(MOD_PATH, 'image')), listdir(join(MOD_PATH, 'mask')))
    
    
def visualize(compute=False):
    load = lambda x: sitk.GetArrayFromImage(sitk.ReadImage(x))
    ct = load("/data/dataset/cmu/v2/image/Dataset000CMU_00004_0000.nii.gz")
    totalseg = load("/data/dataset/cmu/v2/mask/Dataset000CMU_00004_0000.nii.gz")
    """b, v = get_background(totalseg, ct)"""
    if compute:
        vesselness = frangi(window_norm(ct) * 1000)
        v = (vesselness > 0.5 * vesselness.max()).astype(np.uint8)
        np.save("/data/vesselness.npy", vesselness)
    else:
        vesselness = np.load("/data/vesselness.npy")
        v = (vesselness > 0.1 * vesselness.max()).astype(np.uint8)
    save_nifti(v, "/data/ct.nii.gz", None)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cmu", required=False)
    args = parser.parse_args()
    
    """if args.dataset == "cmu":
        preprocess_cmu_split2("/data/dataset/cmu/split2/varname2path.json")
    elif args.dataset == "totalseg":
        # preprocess_totalseg()
        pass
    else:
        raise NotImplementedError(f"[ERROR] dataset {args.dataset} is not valid")"""
    visualize()
