import os
import nibabel
import numpy as np
import pandas as pd
import SimpleITK as sitk

from tqdm import tqdm
from os.path import *
from VesselGen.funcbase import Box, timeit, save_nifti, within_bounds, find_largest_connected_components


def process_(foldername):
    if exists(join(foldername, 'colon_sections.nii.gz')):
        os.remove(join(foldername, 'colon_sections.nii.gz'))
    if exists(join(foldername, 'colon_sections_pca.nii.gz')):
        return 'pca'
    if exists(join(foldername, 'colon_sections_bbox.nii.gz')):
        return 'bbox'

    anchor = sitk.ReadImage(join(foldername, 'segmentations', 'colon.nii.gz'))
    try:
        colon_proj = sitk.GetArrayFromImage(sitk.ReadImage(join(foldername, 'ct_scribble.nii.gz')))
    except RuntimeError:
        colon_proj = nibabel.load(join(foldername, 'ct_scribble.nii.gz')).dataobj[:].transpose(2, 1, 0)  # translate from orthogonal to LPI (zyx) format
    colon_fine_masks, m = process_colon(colon_proj, anchor, [len(os.listdir(join(foldername, 'segmentations'))) + _ for _ in range(6)])
    if m == 'none':
        return m
    save_nifti(colon_fine_masks, join(foldername, f'colon_sections_{m}.nii.gz'), anchor)


@timeit()
def process_colon(colon_proj, colon_mask, vac_labels=[104 + _ for _ in range(6)], mode='pca'):
    from sklearn.decomposition import PCA
    from skimage.morphology import skeletonize_3d
    from scipy.ndimage import binary_dilation, label
    
    flag = False
    colon_mask_c = colon_mask.copy()
    major_parts = find_largest_connected_components(colon_mask_c, p=0.1)
    colon_mask_c[major_parts == 0] = 0
    n_major_parts = major_parts.max()
    if n_major_parts > 10:
        print('retreat to bbox due to inconsistent colon labeling')
        mode = 'bbox'
        
    box = Box((colon_proj > 0) | (colon_mask_c > 0), 5, return_slice=True)
    search_space_proj = colon_proj[*box.cropped_slice]
    search_space_mask = colon_mask_c[*box.cropped_slice]
    search_space_parts = major_parts[*box.cropped_slice]
    
    skeleton = np.zeros(search_space_mask.shape, dtype=np.uint8)
    for i_major_part in range(1, n_major_parts + 1):
        skeleton[skeletonize_3d(search_space_parts == i_major_part) > 0] = i_major_part
        
    if skeleton.sum() < 20:
        mode = 'bbox'
        print('retreat to bbox due to insufficient skeleton points')
    
    if mode == 'pca':
        pca = PCA(n_components=1)
        all_part_skl_anchor_pt = [tuple(np.argwhere(skeleton == i)[0]) for i in range(1, n_major_parts + 1)]
        direction = np.zeros(skeleton.shape + (3,), dtype=np.float32)
        
        for i_major_part in range(n_major_parts):
            dist = -np.ones(skeleton.shape, dtype=int)
            i_part_skl_pts = [all_part_skl_anchor_pt[i_major_part]]
            dist[*i_part_skl_pts[0]] = 0
            
            while len(i_part_skl_pts) > 0:
                x, y, z = i_part_skl_pts.pop()
                sur_skl = np.argwhere(skeleton[x-1: x+2, y-1: y+2, z-1: z+2] > 0) - 1
                for pt in sur_skl:
                    x_, y_, z_ = pt
                    x_ += x
                    y_ += y
                    z_ += z
                    if dist[x_, y_, z_] >= 0: continue
                    
                    dist[x_, y_, z_] = dist[x, y, z] + 1
                    i_part_skl_pts.append((x_, y_, z_))
                    
            start = tuple(np.argwhere(dist == dist.max())[0])
            dist = -np.ones(skeleton.shape, dtype=int)
            dir_cum_coeff = .5
            dir_wgt_coeff = .5
            i_part_skl_pts = [start]
            dist[i_part_skl_pts[0]] = 0
            
            while len(i_part_skl_pts) > 0:
                x, y, z = i_part_skl_pts.pop()
                sur_skl = np.argwhere(skeleton[x-1: x+2, y-1: y+2, z-1: z+2] > 0) - 1
                for pt in sur_skl:
                    x_, y_, z_ = pt
                    x_ += x
                    y_ += y
                    z_ += z
                    if dist[x_, y_, z_] >= 0: continue
                    direction[x_, y_, z_] = dir_cum_coeff * direction[x, y, z] + (1 - dir_cum_coeff) * pt
                    
                    dist[x_, y_, z_] = dist[x, y, z] + 1
                    i_part_skl_pts.append((x_, y_, z_))
                
            direction[dist > 0] /= np.linalg.norm(direction[dist > 0], axis=-1)[..., np.newaxis]
            
        direction[np.isnan(direction)] = 0
        trans_centers = []
        
        for colon_part_idx in range(1, 6):
            raw = search_space_proj == colon_part_idx
            if raw.sum() < 20:  # invalid pred
                continue
            
            pc = np.argwhere(raw)
            pc_mu = np.mean(pc, axis=0)
            pc_fxa = pca.fit(pc).components_[0]
            
            sim = np.abs(np.dot(direction, pc_fxa))   # directional awareness
            sim_skl_p = np.argwhere(sim > 0.5)
            if sim_skl_p.shape[0] == 0:
                mode = 'none'
                print('no valid colon segment deemed similar to scribble orientation, exiting ...')
                return colon_mask, mode
            
            sim_dst_p = np.linalg.norm(sim_skl_p - pc_mu[np.newaxis], axis=1)
            sim_rec_p = -dir_wgt_coeff * sim_dst_p / sim_dst_p.max() + (1 - dir_wgt_coeff) * sim[*sim_skl_p.T.tolist()]
            pc_trans_ctr = sim_skl_p[sim_rec_p.argmax()]
            trans_centers.append(pc_trans_ctr)
            
            pc_ = pc + np.round(pc_trans_ctr - pc_mu).astype(int)
            vox = np.zeros(search_space_mask.shape, dtype=np.uint8)
            # vox[*np.clip(pc_, (0, 0, 0), np.asarray(search_space_mask.shape[::-1])-1).T.tolist()] = vac_labels[colon_part_idx]
            # within_bounds(vox, pc, assign=True, assign_value=2*vac_labels[colon_part_idx])
            within_bounds(vox, pc_, assign=True, assign_value=vac_labels[colon_part_idx])
            
            # search_space_mask[vox > 0] *= vox[vox > 0]
            offset = 1
            search_space_mask[pc_trans_ctr[0] - offset: pc_trans_ctr[0] + offset + 1,
                            pc_trans_ctr[1] - offset: pc_trans_ctr[1] + offset + 1,
                            pc_trans_ctr[2] - offset: pc_trans_ctr[2] + offset + 1,] = vac_labels[colon_part_idx]
    
    if mode == 'bbox':
        
        for colon_part_idx in range(1, 6):
            raw = search_space_proj == colon_part_idx
            if raw.sum() < 20:  # invalid pred
                continue
            
            part_bbox = Box(raw, 1, return_slice=True)
            search_space_mask[*part_bbox.cropped_slice] *= vac_labels[colon_part_idx]
            
        colon_mask_c[*box.cropped_slice] = search_space_mask
        colon_mask_c[(colon_mask_c > 0) & ((colon_mask_c < vac_labels[1]) | (colon_mask_c > vac_labels[-1]))] = 1
        box = Box(colon_mask_c, 0, return_ndarray=True, return_slice=True)
        search_space_mask = box.cropped_array
                
    skip_indices = []
    while (search_space_mask == 1).sum() > 0:
        if len(skip_indices) == 5:
            break
        
        for colon_part_idx in range(1, 6):
            if colon_part_idx in skip_indices:
                continue
            raw = search_space_mask == vac_labels[colon_part_idx]
            dil = binary_dilation(raw, iterations=2, mask=search_space_mask == 1)
            if dil.sum() == raw.sum():
                skip_indices.append(colon_part_idx)
                continue
            if len(skip_indices) == 4:
                search_space_mask[search_space_mask == 1] = vac_labels[colon_part_idx]
            search_space_mask[dil] = vac_labels[colon_part_idx]
        
    ssp = np.zeros(search_space_mask.shape, dtype=np.uint8)
    for colon_part_idx in range(1, 6):
        mask_colon_part_i = find_largest_connected_components(search_space_mask == vac_labels[colon_part_idx], n=1)
        ssp[mask_colon_part_i == 1] = vac_labels[colon_part_idx]
        if (ssp == vac_labels[colon_part_idx]).sum() != (search_space_mask == vac_labels[colon_part_idx]).sum():
            flag = True
            ssp[(search_space_mask == vac_labels[colon_part_idx]) & (mask_colon_part_i == 0)] = 1
    
    ssp_size = (ssp > 0).sum()
    search_space_size = (search_space_parts > 0).sum()
    assert ssp_size >= search_space_size and ssp_size <= search_space_size + (5 * (2 * offset + 1) ** 3) / 2, \
        f"ssp size {ssp_size} does not match search space size {search_space_size}"
    if ssp_size > search_space_size:
        ssp[search_space_parts == 0] = 0
        assert search_space_size == ssp_size, f"ssp size and search space size still not match"
            
    search_space_mask = ssp
    skip_indices = []
    if flag:
        while (search_space_mask == 1).sum() > 0:
            if len(skip_indices) == 5:
                break
            
            for colon_part_idx in range(1, 6):
                if colon_part_idx in skip_indices:
                    continue
                raw = search_space_mask == vac_labels[colon_part_idx]
                dil = binary_dilation(raw, iterations=1, mask=search_space_mask == 1)
                if dil.sum() == raw.sum():
                    skip_indices.append(colon_part_idx)
                    continue
                search_space_mask[dil] = vac_labels[colon_part_idx]
    
    colon_mask_c[*box.cropped_slice] = search_space_mask
        
    return colon_mask_c, mode


def main():
    from VesselGen.macros import BASE_PATH, MOD_PATH
    
    df = pd.read_csv(join(BASE_PATH, 'meta.csv'))
    # foldernames = [df.values[i][0].split(';')[0] for i in range(len(df.values)) if 'abdomen' in df.values[i][0]]
    with open(join(MOD_PATH, 'valid_cases.txt'), 'r') as fp:
        foldernames = [_.strip() for _ in fp.readlines()]
    dummy_foldernames = ['s0010']
        
    # np.random.shuffle(foldernames)
    oks = []
    for foldername in tqdm(foldernames):
        m = process_(join(BASE_PATH, foldername))
        if m == 'pca':
            oks.append(foldername)
    with open(join(MOD_PATH, 'valid_cases.txt'), 'w') as fp:
        fp.writelines([_ + '\n' for _ in oks])
        
        
if __name__ == '__main__':
    main()
    