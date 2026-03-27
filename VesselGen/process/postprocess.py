import numpy as np
import scipy.ndimage as nd
import os, sys
from VesselGen.funcbase import find_largest_connected_components
from skimage.morphology import skeletonize_3d as soft_skeleton3d
import SimpleITK as sitk


def postprocess(mask):
    mask_foreground = mask > 0
    temp = soft_skeleton3d(mask_foreground).astype(np.uint8)
    visits = np.zeros_like(temp)
    visits[temp > 0] = 1
    
    def has_valid_surrounding(pt):
        local = visits[pt[0]-1:pt[0]+2,pt[1]-1:pt[1]+2,pt[2]-1:pt[2]+2]
        if local.sum() > 0:
            valids = np.argwhere(local > 0)
            choice = valids[*np.random.choice(np.arange(len(valids)))]
            return pt - np.array([1, 1, 1]) + choice
        return None
    
    while visits.sum() > 0:
        valids = np.argwhere(visits > 0)
        choice = np.random.choice(np.arange(len(valids)))
        c = valids[choice]

        while c is not None:
            visits[*c] = 0
            d = has_valid_surrounding(c)
            if d is not None: c = d
            else: break
            
        temp[*c] = 2
        
            
    temp = nd.binary_dilation(temp, iterations=3)
    
    return temp
