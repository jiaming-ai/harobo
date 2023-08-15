from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from typing import Tuple
import torch
from torch.nn import functional as F

import numpy as np
import math

from .semantic_annotation import obj_cls_to_id, id_to_obj_cls
from .semantic_annotation import TOP_DOWN_MAP_COLORS, SEMANTIC_MAP_COLORS,META_OBJECTS

from habitat.utils.visualizations import fog_of_war, maps
from PIL import Image, ImageDraw
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)

from utils.transformation import get_affine_grid
from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    visualize_pred,
    save_img_tensor)
# from utils.logging import get_logger
import magnum as mn


def draw_layer_on_map(map, layer, color_idx=1, alpha=128):
    """
    Mix layer on map
    TODO should only mix with foreground with objects
    Parameters:
        map: RGB map
        layer: ndarray of size: [m,n]
    """
    m, n = map.shape[:2]

    foreground = np.zeros([m, n, 4])
    foreground[:, :, :3] = SEMANTIC_MAP_COLORS[layer*color_idx]
    foreground[:, :, 3] = alpha  # 0.5

    # Alpha blending
    map = (
        map.astype(np.int32) * (255 - foreground[:, :, [3]])
        + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
    ) // 255

    return map.astype(np.uint8)


class GTSM:
    """ Ground truth semantic map for training BeaM """

    def __init__(self, sim: HabitatSim, args, gpu_id=None) -> None:
        # we can try different resolution later, by down sampling
        self._map_res = args.map_resolution  # cm per pixel
        self._map_res /= 100  # m per pixel

        self._args = args
        self._sim = sim
        self._sgt = None
        self._map_dim = None  # map of dim C x M x M
        self._map = None  # gt occupancy map with same resolution
        self._scene_id = None
        self._height = None
        self._object_class_num = args.num_object_class

        self._lower_bound = None  # bounds in sim
        self._upper_bound = None
        # args.sim_gpu_id
        self._device = f'cuda:{gpu_id}' if gpu_id is not None else 'cuda:0'

        self._init_sgt()
        self._create_from_sim()
        # logger.info(f'GTSM successfully created!')

    @property
    def map(self):
        return self._map

    def refresh(self):
        """
        Re-build the gtsm if necessary
        Returns true if reset
        """
        if self._is_changed():
            self.reset()
            return True
        else:
            return False

    def reset(self):
        self._init_sgt()
        self._create_from_sim()

    def _is_changed(self):
        """
        Check if the groud truth semantic map has changed due to:
        1. change of the floor
        2. change of the scene
        3. change of objects (do we need this?)
        """
        return self._scene_id != self._sim.config.sim_cfg.scene_id or \
            np.abs(self._height -
                   self._sim.get_agent(0).state.position[1]) > 0.01

    def _create_from_sim(self):
        """
        Create the Ground truth semantic map from the current scene and current floor
        """
        self._scene_id = self._sim.config.sim_cfg.scene_id

        ss = self._sim.semantic_annotations()
        for obj in ss.objects:
            if obj.aabb is None:
                continue
            # if obj.category.name() == 'bag':
            #     print('here')
            center = obj.aabb.center
            if not self._is_on_same_floor(center[1]):
                continue
            idx = obj_cls_to_id(obj.category.name())  # idx for channel index
            if idx == -1:
                continue  # skip obj not covered
            x_len, _, z_len = obj.aabb.sizes / 2.0

            # Nodes to draw rectangle in sim coord in meters
            # we only need the lower left and upper right corners
            corners = [
                center + np.array([x, 0, z])
                for x, z in [
                    (-x_len, -z_len),
                    (x_len, z_len),
                ]
            ]
            # convert to map grid
            map_corners = [self._sim_to_grid(p[2], p[0],) for p in corners]
            self._label_on_map(idx, map_corners)

        # self._sgt[0] = np.bitwise_not(self._map.astype(np.bool))  # negate because true indicates navigable area in original map

    def _label_on_map(self, cid, corners):
        """
        Mark the existence of object cid in the corresponding cells as a rectangle
        """
        (lower_i, lower_j), (upper_i, upper_j) = corners
        self._sgt[cid, lower_i:upper_i, lower_j:upper_j] = 1

    def _sim_to_grid(self, sim_z, sim_x):
        """
        Convert coordinate in simulation to map coordinate.
        The real world coordinates of lower left corner are (coordinate_min, coordinate_min) 
        and of top right corner are (coordinate_max, coordinate_max).

        Lower bound corresponds to (0,0) in the map (which will be rendered upper left in the image)
        """

        grid_size = self._map_res

        # i, frist dimension
        grid_x = int((sim_z - self._lower_bound[2]) / grid_size)
        # j, second dimension
        grid_y = int((sim_x - self._lower_bound[0]) / grid_size)

        return grid_x, grid_y

    def _init_sgt(self):
        r"""
        Init the ground truth semantic map so that it should have the same resolution with the belief semantic map
        """
        pathfinder = self._sim.pathfinder
        self._height = self._sim.get_agent(0).state.position[1]
        res = self._map_res  # m per pixel

        lower_bound, upper_bound = pathfinder.get_bounds()  # in meters

        # map dimension. ignoring the residue (to match with the pathfinder map)
        m, n = [int(abs(upper_bound[coord] - lower_bound[coord]) / res)
                for coord in [2, 0]]
        # map_dim = math.floor(map_dim) + 1
        self._sgt = np.full([self._object_class_num, m, n],
                            0, dtype=bool)  # C x M x M
        self._map_dim = (m, n)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        top_down_map = pathfinder.get_topdown_view(
            meters_per_pixel=res, height=self._height
        ).astype(np.uint8)
        
        if META_OBJECTS is not None:
            ss = self._sim.semantic_annotations()
            for obj in ss.objects:
                if obj.aabb is None:
                    continue
                center = obj.aabb.center
                if not self._is_on_same_floor(center[1]):
                    continue
                if obj.category.name() in ['carpet']:
                    continue# idx for channel index
                x_len, _, z_len = obj.aabb.sizes / 2.0

                # Nodes to draw rectangle in sim coord in meters
                # we only need the lower left and upper right corners
                corners = [
                    center + np.array([x, 0, z])
                    for x, z in [
                        (-x_len, -z_len),
                        (x_len, z_len),
                    ]
                ]
                # convert to map grid
                map_corners = [self._sim_to_grid(p[2], p[0],) for p in corners]

                (lower_i, lower_j), (upper_i, upper_j) = map_corners
                top_down_map[lower_i:upper_i, lower_j:upper_j] = 0 # 1 is traversable

        self._map = top_down_map

    def _is_on_same_floor(self, height,  ceiling_height=2.0):

        # return self._height <= height < self._height + ceiling_height
        # for igibson scenes it's always same floor
        return True 

    def get_agent_polar_angle(self):
        """
        Get the current agent facing direction in eular angle in radius
        This is the angle between the robots heading vector and z-axis in the world coordinate system.
        The grid map transposed x,z axis, meaning the first axis of the grid map is the oringal z axis
        So this angle becomes the poloar angle in the transposed grid map.
        """
        agent_state = self._sim.get_agent(0).state
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
    
    @staticmethod
    def convert_polar_angle_to_world_rot(phi):
        """The opposite of get_agent_polar_angle

        Args:
            phi (_type_): return mn.quaternion in world frame
        """
        z_neg_z_flip = np.pi
        phi = phi - z_neg_z_flip
        quat = mn.Quaternion.rotation(mn.Rad(phi),mn.Vector3([0,1,0]))
        return quat
    
    def convert_map_pos_to_world_pos(self,pos):
        """
        Convert the map position to world position
        """
        
        grid_size = self._map_res
        sim_z = pos[0] * grid_size + self._lower_bound[2]
        sim_x = pos[1] * grid_size + self._lower_bound[0]

        return np.array([sim_x, self._height, sim_z])


    def _get_agent_map_pos(self):
        pos = self._sim.get_agent(0).state.position
        return np.array(self._sim_to_grid(pos[2], pos[0]))

    def _get_vr_mask(self, pos=None, rot=None):
        """
        Return the mask defined by a rectangle where the long edge is the agent's vision range
        use agent's current pos if pos and rot is None
        """
        if pos is None:
            pos_mn = self._get_agent_map_pos()  # the first axis is m
            pos = pos_mn[0], pos_mn[1]
            angle = self.get_agent_polar_angle()
        else:
            pos_mn = self._sim_to_grid(pos[2], pos[0])
            ref_rotation = rot

            heading_vector = quaternion_rotate_vector(
                ref_rotation.inverse(), np.array([0, 0, -1])
            )

            phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
            z_neg_z_flip = np.pi
            angle = np.array(phi) + z_neg_z_flip

        vr = self._args.vision_range
        b = vr / 2

        p1 = pos + b * \
            np.array([np.cos(angle - np.pi/2), np.sin(angle - np.pi/2)])
        p2 = pos + b * \
            np.array([np.cos(angle + np.pi/2), np.sin(angle + np.pi/2)])
        p3 = p1 + vr * np.array([np.cos(angle), np.sin(angle)])
        p4 = p2 + vr * np.array([np.cos(angle), np.sin(angle)])

        m, n = self._map_dim
        img = Image.new('L', (n, m), 0)  # PIL uses w x h as size
        # lower-left, upper-left, upper-right, lower-right
        polygon = [(p[1], p[0]) for p in [p1, p3, p4, p2, p1]]
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        mask = np.array(img)  # agent's fov is denoted as 1

        return mask

    def _get_agent_vision_mask(self):
        """
        Return a mask indicating what the agent currently can see
        """
        mask = np.zeros_like(self._map)
        mask = fog_of_war.reveal_fog_of_war(
            top_down_map=self._map,
            current_fog_of_war_mask=mask,
            current_point=self._get_agent_map_pos(),
            current_angle=self.get_agent_polar_angle(),
            fov=self._args.hfov,
            max_line_len=200,
        )

        return mask

    def get_local_gt_map(self, to_numpy=False):
        """
        Get the ground truth labels of local map using bilinear interpolation
        """
        with torch.no_grad():
            vr = self._args.vision_range
            gtsm_tensor = torch.from_numpy(self._sgt).unsqueeze(
                0).to(torch.float32).to(self._device)

            scale_x = vr / self._map_dim[0]
            scale_y = vr / self._map_dim[1]
            scale_tensor = torch.tensor(
                [scale_y, scale_x], dtype=torch.float32, device=self._device)

            rotate_angle = self.get_agent_polar_angle()
            angle = - rotate_angle  # because of flipping z?
            rot_tensor = torch.tensor(
                [angle], dtype=torch.float32, device=self._device)

            pos = self._get_agent_map_pos()
            normalize_x = self._map_dim[0] / 2
            normalize_y = self._map_dim[1] / 2
            translation_x = (pos[0] + np.cos(rotate_angle)
                             * vr/2 - normalize_x) / normalize_x
            translation_y = (pos[1] + np.sin(rotate_angle)
                             * vr/2 - normalize_y) / normalize_y
            translation_tensor = torch.tensor(
                [translation_y, translation_x], dtype=torch.float32, device=self._device)

            grid = get_affine_grid(translation_tensor,
                                   rot_tensor,
                                   scale_tensor,
                                   [1, self._object_class_num, vr, vr],
                                   self._device
                                   )
            local_bsm = F.grid_sample(gtsm_tensor, grid, align_corners=False)
            if to_numpy:
                return local_bsm[0].cpu().numpy()
            else:
                return local_bsm[0]

    def _draw_st_grid(self, grid):
        """
        Draw affine grid on map
        Parameters:
            grid: s x s x 2
        """
        m, n = self._map_dim
        img = Image.new('L', (n, m), 0)  # PIL uses w x h as size

        def convert(p):
            return (p + 1)/2 * np.array([n, m])

        p1 = convert(grid[0, 0])
        p2 = convert(grid[0, -1])
        p3 = convert(grid[-1, -1])
        p4 = convert(grid[-1, 0])
        # lower-left, upper-left, upper-right, lower-right
        polygon = [(p[0], p[1]) for p in [p1, p2, p3, p4, p1]]
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)

        mask = np.array(img)  # agent's fov is denoted as 1
        return mask

    def get_prior_probabilities(self):
        """
        Get the prior probabilities of each class
        """
        vr = self._args.vision_range
        m, n = self._map_dim
        prior = self._sgt.sum(axis=(1, 2)) / ((m+vr) * (n+vr))
        return prior
    
    def get_local_gt_zmap(self, to_numpy=False):
        """
        Get the local ground truth labels of local z using bilinear interpolation
        """
        with torch.no_grad():
            gtsm_tensor = torch.from_numpy(self._sgt).unsqueeze(
                0).to(torch.float32).to(self._device)
            scale_x = self._args.local_z_size / self._map_dim[0]
            scale_y = self._args.local_z_size / self._map_dim[1]
            scale_tensor = torch.tensor(
                [scale_y, scale_x], dtype=torch.float32, device=self._device)

            angle = - self.get_agent_polar_angle()
            rot_tensor = torch.tensor(
                [angle], dtype=torch.float32, device=self._device)

            pos = self._get_agent_map_pos()
            normalize_x = self._map_dim[0] / 2
            normalize_y = self._map_dim[1] / 2
            translation_x = (pos[0] - normalize_x) / normalize_x
            translation_y = (pos[1] - normalize_y) / normalize_y
            translation_tensor = torch.tensor(
                [translation_y, translation_x], dtype=torch.float32, device=self._device)

            grid = get_affine_grid(translation_tensor,
                                   rot_tensor,
                                   scale_tensor,
                                   [1, self._object_class_num,
                                       self._args.local_z_size, self._args.local_z_size],
                                   self._device
                                   )
            local_bsm = F.grid_sample(
                gtsm_tensor, grid, align_corners=False)[0]

            if to_numpy:
                return local_bsm.cpu().numpy()
            else:
                return local_bsm

    def get_gt_bsm(self, to_numpy=False, pos=None, angle=None):
        """
        Get the ground truth bsm
        using the current robot's pos and rot so that it is at the origin and facing (1,0).
        Note this should be called at the begining of an episode and store this for later use in the same episode
        """
        bsm_dim = self._args.map_size_cm // self._args.map_resolution

        with torch.no_grad():

            buffer = torch.zeros([1, self._object_class_num, bsm_dim,
                                 bsm_dim], dtype=torch.float32, device=self._device)
            m, n = self._map_dim
            x1 = (bsm_dim - m) // 2
            x2 = x1 + m
            y1 = (bsm_dim - n) // 2
            y2 = y1 + n

            if pos is not None:
                pos = np.array(self._sim_to_grid(pos[2], pos[0]))
                angle = - angle
            else:
                pos = self._get_agent_map_pos()
                angle = - self.get_agent_polar_angle()

            sgt = torch.from_numpy(self._sgt)
            # sgt[:,pos[0],pos[1]] = 1
            buffer[0, :, x1:x2, y1:y2] = sgt
            # save_img_tensor(buffer[0,7],'gt_buffer_bsm')

            scale_tensor = torch.tensor(
                [1, 1], dtype=torch.float32, device=self._device)

            rot_tensor = torch.tensor(
                [angle], dtype=torch.float32, device=self._device)

            normalize_x = bsm_dim / 2
            normalize_y = bsm_dim / 2
            # x means first dim, y means second dim
            translation_x = (pos[0] - m/2) / normalize_x
            translation_y = (pos[1] - n/2) / normalize_y
            translation_tensor = torch.tensor(
                [translation_y, translation_x], dtype=torch.float32, device=self._device)

            grid = get_affine_grid(translation_tensor,
                                   rot_tensor,
                                   scale_tensor,
                                   [1, self._object_class_num, bsm_dim, bsm_dim],
                                   self._device
                                   )
            gt_bsm = F.grid_sample(buffer, grid, align_corners=False)[0]

            # save_img_tensor(gt_bsm[7],'gt_bsm')
            if to_numpy:
                return gt_bsm.cpu().numpy()
            return gt_bsm

    def get_st_map(self, pos, angle, obj, to_numpy=False):
        """
        Get spatial transformed map
        """

        map = self.draw_map(obj_list=[obj])
        bsm_dim = self._args.map_size_cm // self._args.map_resolution

        with torch.no_grad():

            buffer = torch.zeros([1, 3, bsm_dim, bsm_dim],
                                 dtype=torch.float32, device=self._device)
            m, n = self._map_dim
            x1 = (bsm_dim - m) // 2
            x2 = x1 + m
            y1 = (bsm_dim - n) // 2
            y2 = y1 + n

            sgt = torch.from_numpy(map).permute(2, 0, 1)
            pos = np.array(self._sim_to_grid(pos[2], pos[0]))
            # sgt[:,pos[0],pos[1]] = 1
            buffer[0, :, x1:x2, y1:y2] = sgt
            # save_img_tensor(buffer[0,7],'gt_buffer_bsm')

            scale_tensor = torch.tensor(
                [1, 1], dtype=torch.float32, device=self._device)

            angle = - angle
            rot_tensor = torch.tensor(
                [angle], dtype=torch.float32, device=self._device)

            normalize_x = bsm_dim / 2
            normalize_y = bsm_dim / 2
            # x means first dim, y means second dim
            translation_x = (pos[0] - m/2) / normalize_x
            translation_y = (pos[1] - n/2) / normalize_y
            translation_tensor = torch.tensor(
                [translation_y, translation_x], dtype=torch.float32, device=self._device)

            grid = get_affine_grid(translation_tensor,
                                   rot_tensor,
                                   scale_tensor,
                                   [1, self._object_class_num, bsm_dim, bsm_dim],
                                   self._device
                                   )
            gt_bsm = F.grid_sample(buffer, grid, align_corners=False)[0]

            # save_img_tensor(gt_bsm[7],'gt_bsm')
            if to_numpy:
                return gt_bsm.cpu().numpy()
            return gt_bsm

    def draw_map(self,
                 obj_list=['cabinet'],
                 layers=[],
                 draw_agent=True,
                 draw_agent_vr=True,
                 always_horizantal=False,
                 save_img=False):
        """
        Draw a top-down map of the environment.

        Parameters:
        -----------
        obj_list : list, optional
            A list of object categories to display on the map. Default is ['chair'].
        layers : list, optional
            A list of additional layers to draw on the map. Default is an empty list.
        draw_agent : bool, optional
            Whether to draw the agent on the map. Default is True.
        draw_agent_vr : bool, optional
            Whether to draw the agent's visible range on the map. Default is True.
        always_horizantal : bool, optional
            Whether to always rotate the map so that it is horizontally aligned. Default is False.
        save_img : bool, optional
            Whether to save the map as an image file. Default is False.

        Returns:
        --------
        numpy.ndarray
            A 2D numpy array representing the top-down map of the environment.
        """

        map = TOP_DOWN_MAP_COLORS[self._map]
        for obj_name in obj_list:
            idx = obj_cls_to_id(obj_name)
            map = draw_layer_on_map(map, self._sgt[idx])

        if draw_agent_vr:
            mask = self._get_vr_mask()
            map = draw_layer_on_map(map, mask)

        # draw additional layers on map if provided
        for l in layers:
            map = draw_layer_on_map(map, l)

        if draw_agent:
            map = maps.draw_agent(
                image=map,
                agent_center_coord=self._get_agent_map_pos(),
                agent_rotation=self.get_agent_polar_angle(),
                agent_radius_px=min(map.shape[0:2]) // 32,
            )

        # rotate 90 degree so it matches the gt map
        if always_horizantal and map.shape[0] > map.shape[1]:
            map = np.rot90(map, 1)

        if save_img:
            save_image(map, 'visualize_map')

        return map

    def get_nearby_objects(self):
        """
        Returns a list of objects near the robot based on the local ground truth map.

        :return: A list of object categories that are near the robot.
        :rtype: list
        """
        local_zmap = self.get_local_gt_zmap(to_numpy=True)
        obj_idx = np.nonzero(np.any(local_zmap, (1, 2)))[0]
        objs = [id_to_obj_cls(obj_idx[i]) for i in range(len(obj_idx))]
        return objs
