import copy
import numpy as np
import torch
from torch.optim import Adam

from VesselGen.trace.A_star import astar
from VesselGen.macros import LABEL_MAPPING
from VesselGen.funcbase import Box, save_nifti, get_rot_xyz, get_dir, printf

from collections import defaultdict
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt


class ChildMaker:
    def __init__(self, tau=2.55, n=3, R=1):
        self.tau = tau
        self.R = R
        self.X = torch.randn((3, n), dtype=torch.float32, requires_grad=True)
        self.r = torch.nn.Parameter(torch.ones_like(torch.tensor(self.get_murray_child_r(R, n, equals=True), dtype=torch.float32)), requires_grad=True)
        self.optim = torch.optim.Adam([self.X, self.r], lr=1e-6)

    def get_children(self, l0=None):
        while loss > 1e-3:
            r_ = self.r ** 2 / torch.linalg.norm(self.r ** 2)
            L = self.R ** 2 * l0 @ r_.T + self.X @ (torch.eye(n) - r_ @ r_.T)
            loss = ((((L ** 2).sum(1) - 1) ** 2).max() + ((self.r ** self.tau).sum() - self.R ** self.tau) ** 2) * 1
            print('loss:', loss.item(), end='\r')
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
    
    @staticmethod
    def get_murray_child_r(r_parent, n_child, tau=3, equals=False, des=None):
        if not equals:
            r_children = np.abs(np.random.randn(n_child))
        else:
            r_children = np.ones(n_child, dtype=np.float32)
        if des is not None:
            r_children = np.array(des)
        r_children = (r_children * r_parent ** tau / r_children.sum()) ** (1 / tau)
        return r_children[:, None]

    

class DTracker:
    def __init__(
        self,
        mask,  ## mask for organs (totalseg)
        start, end,
        wmap_kwargs=(('cube', 5), ('poi', 40), ('poi', 40)),
        **func_kwargs
    ):
        
        self.mask = mask
        self.poi = func_kwargs.get('poi', None)
        self.sp = np.array(start, dtype=int)
        self.ep = np.array(end, dtype=int)
        self.hier_path = defaultdict(list)
        self.XYZ_AVG = func_kwargs.get('XYZ_AVG')
        self.spacing = func_kwargs.get('spacing')
        
        self.box = Box((mask >= LABEL_MAPPING['ascendant_colon']) & (mask <= LABEL_MAPPING['rectum']), func_kwargs.get('search_span', 5))
        if self.poi is None: self.poi = (mask >= LABEL_MAPPING['ascendant_colon']) & (mask <= LABEL_MAPPING['rectum'])
        self._poi = self.poi[*self.box.cropped_slice]
        self._mask = self.mask[*self.box.cropped_slice]
        distance_map = distance_transform_edt(self._mask == 0, sampling=self.spacing)
        self.search_mask = (distance_map < func_kwargs.get('avoid_span', 3) / np.mean(self.spacing)).astype(np.uint8)
        
        self._astar = astar(collision_map=distance_map, **func_kwargs)
        self._sp = self.box.transform(self.sp)
        self._ep = self.box.transform(self.ep)
        self.wmap_kwargs = wmap_kwargs
        
    def _construct_Wmap(self, expand_e_as_er=None):
        
        assert np.array(self.ep).ndim == 1
        self.search_space = np.zeros(self.box.cropped_array.shape)
        search_space = np.zeros(self.box.cropped_array.shape)
        search_space[*self._ep] = 1
        
        surface = self._poi & ~binary_erosion(self._poi)
        self._W = []
        
        for end_region in expand_e_as_er:
            
            if isinstance(end_region, np.ndarray):
                search_space[...] = 1
                search_space[end_region[*self.box.cropped_slice]] = 0
            else:
                if end_region[0] == "cube":
                    
                    half_edge = end_region[1]
                    box_ = Box(search_space, half_edge)
                    
                    search_space[...] = 1
                    search_space[*box_.cropped_slice] = 0
                    
                elif end_region[0] == 'poi':
                    
                    max_dist = end_region[1]
                    x = np.linspace(0, self._poi.shape[0]-1, self._poi.shape[0])
                    y = np.linspace(0, self._poi.shape[1]-1, self._poi.shape[1])
                    z = np.linspace(0, self._poi.shape[2]-1, self._poi.shape[2])
                    xyz = np.asarray(np.meshgrid(x, y, z, indexing='ij'))
                    
                    xyz -= self._ep[:, np.newaxis, np.newaxis, np.newaxis]
                    dist_sq = np.einsum('nijk,nijk->ijk', xyz, xyz)
                    bounded_poi = np.argwhere(dist_sq < max_dist ** 2)
                    
                    search_space[...] = 1
                    search_space[*np.argwhere(self._poi).T.tolist()] = 0
                    
                elif end_region[0] == 'int':
                    
                    skl = skeletonize_3d(self._poi).astype(np.float32) / 255
                    dist_to_skl = distance_transform_edt(1 - skl, sampling=self.spacing)
                    grad_surface = np.asarray(np.gradient(dist_to_skl)).transpose(1, 2, 3, 0)
                    grad_surface[~surface] = 0
                    interior_surface = np.argwhere(np.dot(grad_surface, self._ep - self._sp) > 0)
                                    
                    max_dist = end_region[1]
                    interior_dist = np.linalg.norm(interior_surface - self._ep, axis=1)
                    bounded_interior_surface = interior_surface[interior_dist < max_dist]
                    
                    # draw interior graph
                    if False:
                        save_ = (self.poi * 1).astype(np.uint8)
                        save_[*self.box.inv_transform(bounded_interior_surface).T.tolist()] = 2
                        save_nifti(save_, join(TRASH_BIN, 'interior.nii.gz'), self._)
                    
                    search_space[...] = 1
                    search_space[*interior_surface.T.tolist()] = 0

                if len(self._W) > 0:
                    search_space[~surface] = 1
                    
            self._W.append(distance_transform_edt(search_space, sampling=self.spacing))
            
        self.search_space[self._poi] = 2
        return
    
    def _validify_pt(self, pt, outline=5, mask=None, follow_dir=None):
        if mask is None: mask = self.search_mask
        if mask[*pt] == 0: return pt
        
        neighborhood = Box()
        dir_coeff = dist_coeff = 0.5
        bbox = neighborhood.get_lil_box(mask, pt, outline, return_ndarray=False)
        near_mask = mask[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]]
        valid_pts = np.argwhere(near_mask == 0)
        
        if follow_dir is not None:
            if isinstance(follow_dir, list) or follow_dir.ndim > 1:
                follow_dir = get_dir(follow_dir, -1)
            dir_likeliness = np.dot(valid_pts - np.asarray([outline, outline, outline]), follow_dir)
            near_dist = np.linalg.norm(valid_pts - np.asarray([outline, outline, outline]), axis=1)
            sort_key = -dir_coeff * dir_likeliness + dist_coeff * near_dist / near_dist.max()
            candidate = valid_pts[sort_key.argmin()]
        else:
            sort_key = np.linalg.norm(valid_pts - np.asarray([outline, outline, outline]), axis=1)
            candidate = valid_pts[sort_key.argmin()]
        valid_pt = tuple(np.round(candidate + np.asarray([bbox[0], bbox[2], bbox[4]])).astype(int))
        return valid_pt
    
    def _fill_path(self, s_, e_, dir_=None):
        
        dilated_mask = (self._mask).astype(np.uint8)
        # dilated_mask = self._mask
        
        if isinstance(e_, np.ndarray) and e_.ndim > 1:
            # if dilated_mask[*s_] != 0:
            #     s_ = self._validify_pt(s_, mask=dilated_mask)
            p_ = self._a.isolated_run(dilated_mask, s_, e_, start_dir=dir_)
            e_ = p_[-1]
            return e_, p_
        
        # e_ = self._validify_pt(e_, mask=dilated_mask, follow_dir=-dir_)
        
        # if dilated_mask[*s_] != 0:
        #     s_ = self._validify_pt(s_, mask=dilated_mask)
        p_ = self._a.isolated_run(dilated_mask, s_, e_, start_dir=dir_)
        if p_ is None:
            return e_, None
        e_ = p_[-1]
        return e_, p_
    
    def __trace_by_endpoints_v2(self, start, end, direction, this_bif=None, target_sec_pts=None, layer=0):
        
        if this_bif is not None:
            this_bif = this_bif[0]
        else:
            this_bif = {'p': -100, 'n': (-1, 0)}
        if end is None:
            prim_pt = np.array(start)
            _cand = target_sec_pts
            _dist = np.linalg.norm(_cand - prim_pt[np.newaxis], axis=1)
            endpoint_choices = _cand  # [np.argwhere(_dist < self.XYZ_AVG((layer + 1) * 40)).T[0].tolist()]
            endpoint_dst = _dist  # [np.argwhere(_dist < self.XYZ_AVG((layer + 1) * 40)).T[0].tolist()]
            endpoint_ang = np.dot(endpoint_choices - prim_pt, direction) / np.linalg.norm(endpoint_choices - prim_pt)
            endpoint_p = np.exp(endpoint_ang / 8)  # + np.exp(-endpoint_dst / 8)
            next_endpoint_idx = np.random.choice(np.arange(len(endpoint_choices)), p=endpoint_p / endpoint_p.sum())
            # end = self._validify_pt(endpoint_choices[next_endpoint_idx])
            end = endpoint_choices[next_endpoint_idx]
            # modify direction
            direction = direction * 0.8 + (np.array(end) - np.array(start)) * 0.2
            
        dist_from_end_point = np.random.randint(this_bif['p'] - self.XYZ_AVG(5), this_bif['p'] + self.XYZ_AVG(25))
        pilot_path = self._astar(self.search_mask, start, end, direction, away=max(dist_from_end_point, 3 / np.mean(self.spacing)))
        end_direction = get_dir(pilot_path, -1)
        children_maker = ChildMaker(n=this_bif['n'][1], tau=2.55, R=1)
        end_direction = children_maker.get_children(end_direction[np.newaxis])[0]
        new_bifurs = np.random.randint(*this_bif['n'])
        
        subsequent_path = []
        final_pt = np.array(pilot_path[-1])
        if new_bifurs == 0:
            _cand = target_sec_pts
            _dist = np.linalg.norm(_cand - final_pt[np.newaxis], axis=1)
            endpoint_choices = _cand  # [np.argwhere(_dist < self.XYZ_AVG((layer + 1) * 40)).T[0].tolist()]
            endpoint_dst = _dist  # [np.argwhere(_dist < self.XYZ_AVG((layer + 1) * 40)).T[0].tolist()]
            endpoint_ang = np.dot(endpoint_choices - final_pt, end_direction) / np.linalg.norm(endpoint_choices - final_pt)
            endpoint_p = np.exp(endpoint_ang / 8)  # + np.exp(-endpoint_dst / 8)
            next_endpoint_idx = np.random.choice(np.arange(len(endpoint_choices)), p=endpoint_p / endpoint_p.sum())
            # next_endpoint = self._validify_pt(endpoint_choices[next_endpoint_idx])
            next_endpoint = endpoint_choices[next_endpoint_idx]
            subsequent_path = [self._astar(self.search_mask, pilot_path[-1], next_endpoint, end_direction, away=3 / np.mean(self.spacing))]
            self.hier_path[layer].append(self.box.inv_transform(pilot_path).tolist())
            self.hier_path[layer+1].append(self.box.inv_transform(subsequent_path).tolist())
            return
        elif new_bifurs < 0:
            self.hier_path[layer].append(self.box.inv_transform(pilot_path).tolist())
            return
        else:
            dxyz = np.max(target_sec_pts, axis=0) - np.min(target_sec_pts, axis=0)
            dmax = dxyz.argmax()
            prev_sec_axis_pos = 0
            last_sec_axis_pos = target_sec_pts[:, dmax].min() + (target_sec_pts[:, dmax].max() - target_sec_pts[:, dmax].min()) // new_bifurs
            for bif in range(new_bifurs):
                s = prev_sec_axis_pos - np.random.randint(0, self.XYZ_AVG(30))
                e = last_sec_axis_pos + np.random.randint(0, self.XYZ_AVG(10))
                next_sec = np.array([_ for _ in target_sec_pts if _[dmax] >= s and _[dmax] <= e])
                subsequent_path.append(self.__trace_by_endpoints_v2(pilot_path[-1], None, end_direction, this_bif['b'], next_sec, layer+1))
                prev_sec_axis_pos = e
                last_sec_axis_pos += (target_sec_pts[:, dmax].max() - target_sec_pts[:, dmax].min()) // new_bifurs
        self.hier_path[layer].append(self.box.inv_transform(pilot_path).tolist())
    
    def trace_by_endpoints_v2(self, bifur, start_dir=None):
        
        if start_dir is None:
            start_dir = np.asarray([-1., 0., 0.])
        if not isinstance(start_dir, np.ndarray):
            start_dir = np.array(start_dir)
        start_dir = start_dir.astype(np.float32)
        poi = np.argwhere(self._poi & ~binary_erosion(self._poi))
        poi_ = np.asarray([_ for _ in poi if np.linalg.norm(_ - self._ep) < self.XYZ_AVG(150)])
        self.__trace_by_endpoints_v2(self._sp, self._ep, start_dir, bifur, poi_, 0)
        for _bifur in self.hier_path:
            for _path in self.hier_path[_bifur]:
                for p in _path:
                    p = tuple(p)
                    # self.mask[*p] = 254
        printf(f"[INFO] total tracked {sum([len(_) for _ in self.hier_path.values()])} vessels in {len(self.hier_path.keys())} layers")
        return self.hier_path
    
    def trace_by_endpoints(self, _B=None):
        
        bif = [_B]
        self._construct_Wmap(self.wmap_kwargs)
        max_len = [self.XYZ_AVG(80), self.XYZ_AVG(30), 100000]
        bifur_layers = [0]
        self.path = [[tuple(self._sp)]]
        bifur_directions = [np.asarray([-1, 0, 0])]
        
        i_path = 0
        endpoint_choices = np.argwhere(self._W[0] == 0)
        endpoints = [self._validify_pt(endpoint_choices[np.random.choice(np.arange(len(endpoint_choices)))], 10)]
            
        while i_path < len(self.path):
            
            abort_flag = False
            W = self._W[bifur_layers[i_path]]
            b = bif[i_path]
            _path = []
            path = self.path[i_path]
            bifur_direction = bifur_directions[i_path]
            endpoint = endpoints[i_path]
            print(f'############### Vessel {i_path} Layer {bifur_layers[i_path]} ###############', end=' ')
            
            def check_bifur(e, pt):
                e = np.array(e)
                pt = np.array(pt)
                if e.ndim == 1:
                    dist = np.linalg.norm(e - pt)
                else:
                    dist = np.linalg.norm(np.round(np.argwhere(e).mean(axis=0)).astype(int) - pt)
                if b is None: return -1
                for ind, key in enumerate(list(b.keys())):
                    d = b[key]["p"]
                    if (dist > d[0]) and (dist < d[1]):
                        return ind
                else:
                    return -1
            
            while (W[*(waypoint := path.pop())] > 3 or len(path) < 5) and not abort_flag:
                
                if np.linalg.norm(np.array(waypoint) - np.array(endpoint)) <= 1:
                    print("[INFO] aborting: destination reached")
                    path.extend(_path)
                    abort_flag = True
                
                # randomness = np.linalg.norm(bifur_direction) * (np.random.random((3,)) - 0.5) * np.exp(-(len(path) - 30) ** 2 / 256)
                waypoint_, _path = self._fill_path(waypoint, endpoint, dir_=bifur_direction)
                
                if np.linalg.norm(np.array(endpoint) - np.array(waypoint_)) > 5:
                    abort_flag = len(path) > 10
                    if abort_flag:
                        print("[INFO] aborting: endpoint is possibly not reachable after 2 trials")
                        
                waypoint = waypoint_
                
                if _path is None:
                    print("[INFO] aborting: no valid track around start point")
                    break
                # _l = self.get_dir(_path, len(_path))
                # if _l is None: _l = bifur_direction
                
                for ip, p in enumerate(_path):
                        
                    if (bifur_ind := check_bifur(endpoints[i_path], p)) >= 0 or \
                        (abort_flag and (bifur_ind := check_bifur(endpoints[i_path], endpoints[i_path])) >= 0):
                        if np.random.random() < 0.8 and len(path) + ip > 5:
                            n_bifurs = np.random.choice(range(*b[bifur_ind]["n"]))
                            bifur_of_bifurs = b[bifur_ind]["b"]
                            del b[bifur_ind]
                            
                            # get bifur points 
                            for _ in range(n_bifurs):
                                self.path.append([p])
                                bif.append(copy.deepcopy(bifur_of_bifurs))
                                bifur_layers.append(bifur_layers[i_path] + 1)
                                rot_matrix = get_rot_xyz(np.array([np.random.randint(40, 60)] * 3))
                                bifur_directions.append(get_dir(_path, -1))
                                
                                _w = self._W[bifur_layers[i_path] + 1]
                                _cand = np.argwhere(_w == 0)
                                _dist = np.linalg.norm(_cand - self._ep[np.newaxis], axis=1)
                                
                                endpoint_choices = _cand[np.argwhere(_dist < self.XYZ_AVG(40) * (bifur_layers[i_path] + 1)).T[0].tolist()]
                                # endpoint_dist = _dist[np.argwhere(_dist < self.XYZ_AVG(40) * (bifur_layers[i_path] + 1)).T[0].tolist()]
                                endpoint_ang = np.dot(endpoint_choices - _path[ip], get_dir(_path, ip)) / np.linalg.norm(endpoint_choices - _path[ip])
                                endpoint_p = np.exp(endpoint_ang / 8)
                                next_endpoint_idx = np.random.choice(np.arange(len(endpoint_choices)), p=endpoint_p / endpoint_p.sum())
                                endpoints.append(self._validify_pt(endpoint_choices[next_endpoint_idx]))
                                _w[*self.box.get_lil_box(_w, endpoints[-1], 5, return_slice=True)] = -1
                                
                            if len(b) == 0:
                                abort_flag = True
                                print("[INFO] aborting: bifurcation", end=' ')
                                
                            """if len(path) + ip > max_len[bifur_layers[i_path]]:
                                abort_flag = True
                                print("[INFO] maximum vessel length reached", end=' ')"""
                                
                    if abort_flag or ip > 25:
                        path.extend(_path[:ip])
                        break
                        
                    if ip < len(_path) - 1 and not abort_flag:
                        if self._mask[*p]: 
                            d = 1
                        self.search_space[*p] = i_path + 10
                        self._mask[*p] = 255
                        
                else:
                    path.extend(_path)
                    
                waypoint = _path[-1]
                if len(path) > 1:
                    bifur_direction = get_dir(path, -1)
            
            print(f"Tracked Length {len(path)}")
            i_path += 1
        
        # draw intermediate graph
        if False:
            illust_ = np.zeros(self.mask.shape, dtype=np.uint8)
            temp_ = np.zeros(self.mask.shape, dtype=np.uint8)
                
            for ip, p in enumerate(self.path):
                if len(p) != 0:
                    temp_[*self.box.inv_transform(p).T.tolist()] = 1
                    temp_ = binary_dilation(temp_ > 0)
                    illust_[temp_ > 0] = ip + 10
                    temp_[...] = 0
            
            illust_[*self.box.get_lil_box(self.mask, self.sp, 2, True)] = 2
            illust_[*self.box.inv_transform(np.argwhere(self.search_space == 2)).T.tolist()] = 3
            save_nifti(illust_, join(MOD_PATH, 'trash', f"intermediate.nii.gz"), self._)
        
        hier_path = defaultdict(list)
        for ip, p in enumerate(self.path):
            if len(p) != 0:
                hier_path[bifur_layers[ip]].append(self.box.inv_transform(p).tolist())
        return hier_path
        
    def trace_by_extension(self, stride=None, _B=None):
        
        bif = [_B]
        paces = [0]
        bifur_layers = [0]
        self.path = [[tuple(self._sp)]]
        stride = lambda x: max(20 * np.exp(-x / 2), 5)
        bifur_directions = [-(self._ep - self._sp).astype(np.float32) * np.asarray([0, 0, 1])]
        
        i_path = 0
        branches = [100, 30, 1000]
        
        while i_path < len(self.path):
            
            abort_flag = False
            
            ib = 0
            W = self._W[bifur_layers[i_path]]
            b = bif[i_path]
            path = self.path[i_path]
            bifur_direction = bifur_directions[i_path]
            print(f'############### Vessel {i_path} Layer {bifur_layers[i_path]} ###############')
            
            def check_bifur(pt):
                dist = W[*pt]
                if b is None: return -1
                print(dist)
                for ind, key in enumerate(list(b.keys())):
                    d = b[key]["p"]
                    if (dist > d[0]) and (dist < d[1]):
                        return ind
                else:
                    return -1
            
            while (not abort_flag and W[*(waypoint := path.pop())] > 2) or ib == 0:
                
                if bifur_direction is None:
                    x, y, z = waypoint
                    _g = -np.asarray([(W[x + 1, y, z] - W[x - 1, y, z]) / 2,
                                    (W[x, y + 1, z] - W[x, y - 1, z]) / 2,
                                    (W[x, y, z + 1] - W[x, y, z - 1]) / 2,])
                    _l = self.get_dir(path, len(path))
                    next_waypoint = waypoint + _g / np.linalg.norm(_g) * stride(len(path))
                else:
                    _g = bifur_direction
                    _l = bifur_direction
                    _stride = np.arange(5, 10)
                    _p = np.exp(-_stride / 5)
                    _g += (np.random.random((3,)) - 0.5) * np.linalg.norm(_g) * 0.5
                    next_waypoint = waypoint + _g / np.linalg.norm(_g) * np.random.choice(_stride, p=_p / _p.sum())
                
                next_waypoint = tuple(np.clip(np.round(next_waypoint).astype(int), [0, 0, 0], np.array(self._mask.shape) - 1))
                
                if waypoint == next_waypoint:
                    print("[INFO] aborting: small grad magnitude")
                    abort_flag = True
                
                next_waypoint, _path = self._fill_path(waypoint, next_waypoint, dir_=_l)
                bifur_direction = None
                
                for ip, p in enumerate(_path):
                    
                    if abort_flag:
                        path.extend(_path[:ip])
                        break
                    
                    if ib > branches[bifur_layers[i_path]]:
                        print("[INFO] aborting: maximum branch length exceeded")
                        abort_flag = True
                    
                    if (bifur_ind := check_bifur(p)) >= 0:
                        
                        if np.random.random() < 0.8 and ib > 5:
                            n_bifurs = np.random.choice(range(*b[bifur_ind]["n"]))
                            bifur_of_bifurs = b[bifur_ind]["b"]
                            del b[bifur_ind]
                            
                            for _ in range(n_bifurs):
                                bif.append(copy.deepcopy(bifur_of_bifurs))
                                self.path.append([p])
                                bifur_layers.append(bifur_layers[i_path] + 1)
                                rot_matrix = self.get_rot_xyz(np.array([np.random.randint(40, 60)] * 3))
                                bifur_directions.append(rot_matrix @ _l)
                                paces.append(len(path))
                            
                            if len(b) == 0:
                                abort_flag = True
                                print("[INFO] aborting: bifurcation")
                        
                    if ip < len(_path) - 1 and not abort_flag:
                        self.search_space[*p] = i_path + 10
                        self._mask[*p] = 255
                        ib += 1
                        
                else:
                    path.extend(_path)
                
                waypoint = next_waypoint
            
            if waypoint != path[-1]:
                path.append(waypoint)
                self.search_space[*waypoint] = i_path + 10
                self._mask[*waypoint] = 255      
            i_path += 1
        
        illust_ = np.zeros(self.mask.shape, dtype=np.uint8)
        temp_ = np.zeros(self.mask.shape, dtype=np.uint8)
        for ip, p in enumerate(self.path):
            temp_[*self.box.inv_transform(p).T.tolist()] = 1
            temp_ = binary_dilation(temp_ > 0)
            illust_[temp_ > 0] = ip + 10
            temp_[...] = 0
        
        """illust_[*self.box.get_lil_box(self.mask, self.sp, 2, True)] = 2
        illust_[*self.box.inv_transform(np.argwhere(self.search_space == 2)).T.tolist()] = 3
        save_nifti(illust_, os.path.join(MOD_PATH, 'trash', f"intermediate.nii.gz"), self._)"""
        
        hier_path = defaultdict(list)
        for i_path, path in enumerate(self.path):
            hier_path[bifur_layers[i_path]].append(self.box.inv_transform(path).tolist())
        return hier_path