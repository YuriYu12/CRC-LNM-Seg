import numpy as np
from scipy.ndimage import binary_dilation
from VesselGen.funcbase import printf, get_dir, get_rot_xyz
from VesselGen.macros import ADJACENT_NODES_2D, ADJACENT_NODES_3D


class Node:
    __slots__ = 'pos', 'parent', 'f', 'g', 'h', 'd', 'c'
    
    def __init__(self, parent, pos):
        self.pos = pos
        self.parent = parent
        
        self.f = 0
        self.g = 0
        self.h = 0
        self.d = 0
        self.c = 0
        
    def __eq__(self, other):
        return all([(self.pos[_] - other.pos[_]) == 0 for _ in range(len(self.pos))])
    
    def __getitem__(self, index):
        return self.pos[index]


class Astar:
    def __init__(self, map, **kwargs):
        
        self.map = map
        self.path = []
        open_list = []
        close_list = []
        self.ndim = len(self.map.shape)
        self.adjacent_nodes = ADJACENT_NODES_3D if self.ndim == 3 else ADJACENT_NODES_2D
        
        # to what extent to follow a given direction, a higher value results in more time complexity
        # eg. Lambda=0.7 -> to get back to original track, #extra steps taken is log_{0.7}(0.01) = 13
        self.Lambda = kwargs.get('lambda', 0.91)
        self.dist_lb = kwargs.get('distance_lowerbound', 10)
        self.dist_ub = kwargs.get('distance_upperbound', 21)
        
        # avoid organ and/or vessel collision, 
        # 1) maybe needs more iterations for vessels with large radius or ct scans with higher resolutions
        # 2) maybe require special adaptation to anisotrophic scans
        self.map = binary_dilation(self.map > 0, iterations=3).astype(np.uint8)
        
        # compulsory start direction of a vessel tracker
        self.start_direction = kwargs.get('start_direction', None)
        self.waypoint_directions = kwargs.get('waypoint_directions', None)
        
        # check whether tracing has ended in a feasible region / point
        self.start = None
        self.end = None
        self._is_tracing_done = lambda x, y: x == y
        
    def _parse_end_node_or_area(self, end):
        if isinstance(end, tuple):
            self._is_tracing_done = lambda x, y: x == y
            return Node(None, end)
        
        if isinstance(end, np.ndarray):
            if end.ndim == 1:
                self._is_tracing_done = lambda x, y: x == y
                return Node(None, end)
            assert end.shape == self.map.shape
            end_avg_node = Node(None, tuple(np.round(np.argwhere(end > 0).mean(axis=0)).astype(int)))
            self._is_tracing_done = lambda x, _: self.end[*x.pos]
            return end_avg_node
        
    def __binary_search(self, arr, l, r, item):
        if l == r:
            return -1
        m = (l + r) // 2
        if arr[m] == item:
            return m
        
        lfind = self.__binary_search(arr, l, m, item)
        rfind = self.__binary_search(arr, m+1, r, item)
        if lfind != -1: return lfind
        if rfind != -1: return rfind
        return -1
        
    def _binary_search(self, arr, item):
        return self.__binary_search(arr, 0, len(arr), item)
    
    def isolated_run(self, search_space, start, end, start_dir=None, away=0, Lambda=None):
        path = []
        open_list = []
        close_list = []
        if Lambda is not None: self.Lambda = Lambda

        start_node = Node(None, start)
        end_node = self._parse_end_node_or_area(end)
        if start_dir is not None:
            start_dir /= np.linalg.norm(start_dir)
        
        self.start = start
        self.end = end
        open_list.append(start_node)
        
        while len(open_list) > 0:
            current_node: Node = open_list[0]
                    
            open_list.pop(0)
            close_list.append(current_node)
            
            if self._is_tracing_done(current_node, end_node):
                backtrack_node = current_node
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                return path[::-1]
            
            # early stopping to cap search time
            if len(open_list) > 1500 or len(close_list) > 3000:
                best_node = np.array([np.linalg.norm(np.array(p.pos) - np.array(end)) for p in close_list])
                backtrack_node = close_list[best_node.argmin()]
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                printf(f"[WARNING] A-star unable to converge, best try at {close_list[best_node.argmin()].pos}, end is {end_node.pos}")
                return path[::-1]
            
            if current_node.g > 10 and current_node.h < away ** 2:
                backtrack_node = current_node
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                printf(f"[INFO] A-star has found node {current_node.pos}, {np.sqrt(current_node.h):.2f} away from destination {end_node.pos}")
                return path[::-1]
            
            children = []
            for deltapos in self.adjacent_nodes:
                # 1) check whether in the range of `self.mask`
                current_pos = tuple(current_node.pos[_] + deltapos[_] for _ in range(self.ndim))
                if current_pos[0] < 0 or current_pos[0] >= search_space.shape[0] or \
                    current_pos[1] < 0 or current_pos[1] >= search_space.shape[1]:
                    continue
                if self.ndim == 3 and (current_pos[2] < 0 or current_pos[2] >= search_space.shape[2]):
                    continue
                # 2) check whether hit a valid label (organ, auxiliary spheres, etc.)
                if search_space[current_pos] > 0:
                    continue
                # 3) if not, append to candidate points
                children.append(Node(current_node, current_pos))
            
            for child in children:
                for closed_child in close_list:
                    if child == closed_child:
                        break
                else:
                    child.g = current_node.g + 1
                    for index, open_node in enumerate(open_list):
                        if child == open_node:
                            if child.g >= open_node.g:
                                break
                            else:
                                open_list.pop(index)
                    else:
                        index = 0
                        child.h = sum([(child.pos[_] - end_node.pos[_]) ** 2 for _ in range(self.ndim)])
                        if start_dir is not None:
                            real_dir = (np.array(child.pos) - np.array(start_node.pos)).astype(np.float32)
                            real_dir /= np.linalg.norm(real_dir)
                            child.d = -real_dir.dot(start_dir) * (child.g + child.h) * self.Lambda ** current_node.g
                            
                        child.f = child.g + child.h + child.d
                        # insertion sort
                        while index < len(open_list) and open_list[index].f < child.f:
                            index += 1
                        open_list.insert(index, child)
        
    def run(self, start, end, **kwargs):
        self.path = []
        self.start_direction = kwargs.get('start_direction', self.start_direction)
        self.waypoint_directions = kwargs.get('waypoint_directions', self.waypoint_directions)
        search_space = (kwargs.get('search_space', np.full(self.map.shape, fill_value=False)) | (self.map > 0)).astype(np.uint8)
        
        self.start = start
        self.end = end
        start_node = Node(None, start)
        end_node = self._parse_end_node_or_area(end)
        
        if self.waypoint_directions is not None:
            nodes_elapsed, waypoint_dirs = self.waypoint_directions
        
        open_list.append(start_node)
        
        while len(open_list) > 0:
            current_node: Node = open_list[0]
                    
            open_list.pop(0)
            close_list.append(current_node)
            
            if self._is_tracing_done(current_node, end_node):
                backtrack_node = current_node
                while backtrack_node is not None:
                    self.path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                open_list = []
                close_list = []
                return self.path[::-1]
            
            children = []
            for deltapos in self.adjacent_nodes:
                # 1) check whether in the range of `self.mask`
                current_pos = tuple(current_node.pos[_] + deltapos[_] for _ in range(self.ndim))
                if current_pos[0] < 0 or current_pos[0] >= search_space.shape[0] or \
                    current_pos[1] < 0 or current_pos[1] >= search_space.shape[1]:
                    continue
                if self.ndim == 3 and (current_pos[2] < 0 or current_pos[2] >= search_space.shape[2]):
                    continue
                # 2) check whether hit a valid label (organ, auxiliary spheres, etc.)
                if search_space[current_pos] > 0:
                    continue
                # 3) if not, append to candidate points
                children.append(Node(current_node, current_pos))
            
            for child in children:
                for closed_child in close_list:
                    if child == closed_child:
                        break
                else:
                    child.g = current_node.g + 1
                    for index, open_node in enumerate(open_list):
                        if child == open_node:
                            if child.g >= open_node.g:
                                break
                            else:
                                open_list.pop(index)
                    else:
                        index = 0
                        child.h = sum([(child.pos[_] - end_node.pos[_]) ** 2 for _ in range(self.ndim)])
                        
                        # maybe should impose a max limit on #extra nodes
                        if self.start_direction is not None:
                            real_dir = (np.array(child.pos) - np.array(start_node.pos)).astype(np.float32)
                            real_dir /= np.linalg.norm(real_dir)
                            child.d = -real_dir.dot(self.start_direction) * (child.g + child.h) * self.Lambda ** current_node.g
                        
                        if self.waypoint_directions is not None:
                            index = self._binary_search(nodes_elapsed, current_node.g)
                            if index != -1:
                                real_dir = (np.array(child.pos) - np.array(start_node.pos)).astype(np.float32)
                                real_dir /= np.linalg.norm(real_dir)
                                child.d = -real_dir.dot(waypoint_dirs[index]) * (child.g + child.h) * self.Lambda ** current_node.g
                            
                        child.f = child.g + child.h + child.d + child.c
                        # insertion sort
                        while index < len(open_list) and open_list[index].f < child.f:
                            index += 1
                        open_list.insert(index, child)


class ChildMaker:
    def __init__(self, alpha):
        pass


class astar:
    def __init__(self, *args, **kwargs):
        self.away = kwargs.get("away", 0)
        self.Lambda = kwargs.get("Lambda", 0.9)
        self.exit_step = kwargs.get("exit_step", 2500)
        self.preserve_step = kwargs.get("preserve_step", 3)
        
    def __trace(self, search_space, start, end, start_dir=None, **kwargs):
        path = []
        open_list = []
        close_list = []
        away = kwargs["away"]
        Lambda = kwargs["Lambda"]
        exit_step = kwargs["exit_step"]
        preserve_step = kwargs["preserve_step"]
        
        if Lambda is None: Lambda = 0.9
        _is_tracing_done = None
        
        def _parse_end_node_or_area(end):
            nonlocal _is_tracing_done
            if isinstance(end, np.ndarray):
                if end.ndim == 1:
                    _is_tracing_done = lambda x, y: x == y
                    return Node(None, end)
                end_avg_node = Node(None, tuple(np.round(np.argwhere(end > 0).mean(axis=0)).astype(int)))
                _is_tracing_done = lambda x, _: end[*x.pos]
                return end_avg_node
            else:
                _is_tracing_done = lambda x, y: x == y
                return Node(None, end)

        start_node = Node(None, start)
        if start_dir is not None:
            start_dir /= np.linalg.norm(start_dir)
            
        end_node = _parse_end_node_or_area(end)
        open_list.append(start_node)
        
        while len(open_list) > 0:
            current_node: Node = open_list[0]
                    
            open_list.pop(0)
            close_list.append(current_node)
            
            if _is_tracing_done(current_node, end_node):
                backtrack_node = current_node
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                return path[::-1]
            
            # early stopping to cap search time
            if len(close_list) > exit_step:
                all_list = open_list + close_list
                best_node = np.array([np.linalg.norm(np.array(p.pos) - np.array(end_node.pos)) for p in all_list])
                backtrack_node = all_list[best_node.argmin()]
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                printf(f"[WARNING] A-star unable to converge on time, starting at {start_node.pos}, best try at {all_list[best_node.argmin()].pos}, end is {end_node.pos}")
                return path[::-1]
            
            if current_node.g > preserve_step and current_node.h < away ** 2:
                backtrack_node = current_node
                while backtrack_node is not None:
                    path.append(backtrack_node.pos)
                    backtrack_node = backtrack_node.parent
                printf(f"[INFO] A-star has found node {current_node.pos}, {np.sqrt(current_node.h):.2f} away from destination {end_node.pos}")
                return path[::-1]
            
            children = []
            for deltapos in ADJACENT_NODES_3D:
                # 1) check whether in the range of `self.mask`
                current_pos = tuple(current_node.pos[_] + deltapos[_] for _ in range(3))
                if current_pos[0] < 0 or current_pos[0] >= search_space.shape[0] or \
                    current_pos[1] < 0 or current_pos[1] >= search_space.shape[1] or \
                        current_pos[2] < 0 or current_pos[2] >= search_space.shape[2]:
                    continue
                # 2) check whether hit a valid label (organ, auxiliary spheres, etc.)
                if search_space[current_pos] > 0 and current_node.g > preserve_step:
                    continue
                # 3) if not, append to candidate points
                children.append(Node(current_node, current_pos))
            
            for child in children:
                for closed_child in close_list:
                    if child == closed_child:
                        break
                else:
                    child.g = current_node.g + 1
                    for index, open_node in enumerate(open_list):
                        if child == open_node:
                            if child.g >= open_node.g:
                                break
                            else:
                                open_list.pop(index)
                    else:
                        index = 0
                        child.h = sum([(child.pos[_] - end_node.pos[_]) ** 2 for _ in range(3)])
                        if start_dir is not None:
                            real_dir = (np.array(child.pos) - np.array(start_node.pos)).astype(np.float32)
                            real_dir /= np.linalg.norm(real_dir)
                            child.d = -real_dir.dot(start_dir) * (child.g + child.h) * Lambda ** current_node.g
                        if self._collision_map is not None:
                            child.c = -max(child.h, 36) * (1 - self._collision_map[*child.pos])
                            
                        child.f = child.g + child.h + child.d + child.c
                        # insertion sort
                        while index < len(open_list) and open_list[index].f < child.f:
                            index += 1
                        open_list.insert(index, child)
        
        max_step = max([_.g for _ in open_list + close_list])
        printf(f"[ERROR] A-star trapped in a local area around {start_node.pos} after {max_step} steps, preserve_step ++")
        kwargs["preserve_step"] = preserve_step + 1
        if kwargs["preserve_step"] > 12: 
            raise RuntimeError(f"[ERROR] A-star trapped in a local area around {start_node.pos} after {max_step} steps, preserve_step ++")
        return self.__trace(search_space, start, end, start_dir=None, **kwargs)
    
    def __call__(
        self,
        search_space,
        start, end, start_dir=None,
        **kwargs
    ):
        _start = start
        _end = end
        _start_dir = start_dir
        away = kwargs.get('away', self.away)
        Lambda = kwargs.get('Lambda', self.Lambda)
        exit_step = kwargs.get('exit_step', self.exit_step)
        self._collision_map = kwargs.get('collision_map', None)
        preserve_step = kwargs.get('preseve_step', self.preserve_step)
        _kwargs = dict(away=away, Lambda=Lambda, exit_step=exit_step, preserve_step=preserve_step)
        
        if self._collision_map is not None:
            self._collision_map = np.exp(-self._collision_map / 3)
        
        traced_path = []
        single_traced_len = 40
        while True:
            newly_traced_path = self.__trace(search_space, _start, _end, _start_dir, **_kwargs)
            if len(newly_traced_path) > single_traced_len:
                _dir = get_dir(newly_traced_path, -1)
                _rot = get_rot_xyz(np.array([np.random.randint(60, 120) for _ in range(3)]))
                _start_dir = _rot @ _dir
                _start = newly_traced_path[single_traced_len]
                _kwargs["preserve_step"] = 1
                _kwargs["Lambda"] = 0.91
                traced_path.extend(newly_traced_path[:single_traced_len])
                printf(f"[INFO] retraced at {single_traced_len} / {len(newly_traced_path)}")
            else:
                traced_path.extend(newly_traced_path)
                break
            
        return traced_path


if __name__ == '__main__':
    test_map = np.zeros((5, 5))
    test_map[1, :4] = 1
    test_map[3, 2:] = 1
    test_start = (0, 0)
    test_end = (4, 4)

    tracker = astar(test_map)
    print(tracker.run(test_start, test_end))