# Adapted from https://github.com/facebookresearch/home-robot

import numpy as np
import torch

from mapping.map_utils import MapSizeParameters, init_map_and_pose_for_env
from mapping.polo.constants import MapConstants as MC
from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    render_plt_image,
    visualize_pred,
    show_points, 
    show_voxel_with_prob, 
    show_voxel_with_logit,
    save_img_tensor)


def dialate_tensor(tensor, kernel_size=3):
    """
    Dialates a tensor by a given kernel size.
    args:
        tensor: tensor to dialate (C, H, W)
        kernel_size: size of dialation kernel
    """
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32).to(tensor.device)
    return torch.nn.functional.conv2d(
        tensor.unsqueeze(1), kernel, padding=kernel_size // 2
    ).squeeze(1)

class POLoMapState:
    """
   
    """

    def __init__(
        self,
        device: torch.device,
        num_environments: int,
        num_sem_categories: int,
        map_resolution: int,
        map_size_cm: int,
        global_downscaling: int,
        probability_prior: float,
    ):
        """
        Arguments:
            device: torch device on which to store map state
            num_environments: number of parallel maps (always 1 in real-world but
             multiple in simulation)
            num_sem_categories: number of semantic channels in the map
            map_resolution: size of map bins (in centimeters)
            map_size_cm: global map size (in centimetres)
            global_downscaling: ratio of global over local map
        """
        self.device = device
        self.num_environments = num_environments
        self.num_sem_categories = num_sem_categories

        self.map_size_parameters = MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution

        # TODO remove unncecessary channels
        # Map consists of multiple channels (5 NON_SEM_CHANNELS followed by semantic channels) containing the following:
        # 0: Obstacle Map
        # 1: Explored Area
        # 2: Current Agent Location
        # 3: Past Agent Locations
        # 4: Regions agent has been close to
        # 5, 6, 7, .., num_sem_categories + 5: Semantic Categories
        num_channels = self.num_sem_categories + MC.NON_SEM_CHANNELS # voxel height
        # num_channels = self.num_sem_categories + MC.NON_SEM_CHANNELS

        self.global_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.global_map_size,
            self.global_map_size,
            device=self.device,
        )
        self.local_map = torch.zeros(
            self.num_environments,
            num_channels,
            self.local_map_size,
            self.local_map_size,
            device=self.device,
        )

        # Global and local (x, y, o) sensor pose
        # This is in the hab world frame (x: forward, y: left, z: up)  unit: meter
        # note: this is not the same as GPS
        # global_pose = gps + map_size_cm/2/100
        self.global_pose = torch.zeros(self.num_environments, 3, device=self.device)
        # always be (12, 12) since the local map is centered at the agent
        self.local_pose = torch.zeros(self.num_environments, 3, device=self.device)

        # Origin of local map (3rd dimension stays 0)
        # This is in the hab world frame (x: forward, y: left, z: up)  unit: meter
        self.origins = torch.zeros(self.num_environments, 3, device=self.device)

        # Local map boundaries
        # map frame: x: right, y: down

        self.lmb = torch.zeros(
            self.num_environments, 4, dtype=torch.int32, device=self.device
        )

        # Binary map encoding agent high-level goal
        self.goal_map = np.zeros(
            (self.num_environments, self.local_map_size, self.local_map_size)
        )
        self.frontier_map = np.zeros(
            (self.num_environments, self.local_map_size, self.local_map_size)
        )

        self.prior = probability_prior
        self.prior_logit = torch.logit(torch.tensor(self.prior))
        self.local_coords = np.array([self.local_map_size // 2, self.local_map_size // 2])

    
    def init_map_and_pose(self):
        """Initialize global and local map and sensor pose variables."""
        for e in range(self.num_environments):
            self.init_map_and_pose_for_env(e)

    def init_map_and_pose_for_env(self, e: int):
        """Initialize global and local map and sensor pose variables for
        a specific environment.
        """
        init_map_and_pose_for_env(
            e,
            self.local_map,
            self.global_map,
            self.local_pose,
            self.global_pose,
            self.lmb,
            self.origins,
            self.map_size_parameters,
        )
        self.goal_map[e] *= 0.0

        # Set probability to priors
        self.global_map[:, MC.PROBABILITY_MAP, :, :] = torch.logit(torch.tensor(self.prior))
        self.local_map[:, MC.PROBABILITY_MAP, :, :] = torch.logit(torch.tensor(self.prior))
        
        self.global_map[:, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :] = -torch.inf # marks as empty
        self.local_map[:, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :] = -torch.inf # marks as empty

    def update_frontier_map(self, e: int, frontier_map: np.ndarray):
        """Update the current exploration frontier."""
        self.frontier_map[e] = frontier_map

    def get_frontier_map(self, e: int):
        return self.frontier_map[e]

    def update_global_goal_for_env(self, e: int, goal_map: np.ndarray):
        """Update global goal for a specific environment with the goal action chosen
        by the policy.

        Arguments:
            goal_map: binary map encoding goal(s) of shape (batch_size, M, M)
        """
        self.goal_map[e] = goal_map

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------
    
    def get_obstacle_map(self, e) -> np.ndarray:
        """Get local obstacle map for an environment."""
        return np.copy(self.local_map[e, MC.OBSTACLE_MAP, :, :].cpu().float().numpy())

    def get_explored_map(self, e) -> np.ndarray:
        """Get local explored map for an environment."""
        return np.copy(self.local_map[e, MC.EXPLORED_MAP, :, :].cpu().float().numpy())

    def get_visited_map(self, e) -> np.ndarray:
        """Get local visited map for an environment."""
        return np.copy(self.local_map[e, MC.VISITED_MAP, :, :].cpu().float().numpy())

    def get_been_close_map(self, e) -> np.ndarray:
        """Get map showing regions the agent has been close to"""
        return np.copy(self.local_map[e, MC.BEEN_CLOSE_MAP, :, :].cpu().float().numpy())

    def get_semantic_map(self, e) -> np.ndarray:
        """Get local map of semantic categories for an environment."""
        semantic_map = np.copy(self.local_map[e].cpu().float().numpy())
        semantic_map[
            MC.NON_SEM_CHANNELS + self.num_sem_categories - 1, :, :
        ] = 1e-5  # Last category is unlabeled
        semantic_map = semantic_map[
            MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + self.num_sem_categories, :, :
        ].argmax(0)
        return semantic_map

    def get_local_coords(self, e) -> np.ndarray:
        """Get local coordinates for an environment."""
        return np.copy(self.local_coords)
    
    def get_probability_map_normalized(self, e) -> np.ndarray:
        """Get local probability map for an environment."""
        prob_map = torch.sigmoid(self.local_map[e, MC.PROBABILITY_MAP, :, :])
        prob_map = prob_map / prob_map.sum()
        
        return np.copy(prob_map.cpu().numpy())
    
    def get_probability_map(self, e) -> np.ndarray:
        """Get local probability map for an environment."""
        prob_map = torch.sigmoid(self.local_map[e, MC.PROBABILITY_MAP, :, :])
        
        return np.copy(prob_map.cpu().numpy())
    
    def get_global_probability_map_tensor(self, e) -> np.ndarray:
        """Get local probability map for an environment."""
        return torch.sigmoid(self.global_map[e, MC.PROBABILITY_MAP, :, :]).clone()
        
    def get_global_exp_map_tensor(self, e) -> np.ndarray:
        """Get local probability map for an environment."""
        return (self.global_map[e, MC.EXPLORED_MAP, :, :]).clone()

    def get_promising_map(self, e) -> np.ndarray:
        """Get local promising map for an environment."""
        prob_map = torch.sigmoid(self.local_map[e, MC.PROBABILITY_MAP, :, :])
        
        return np.copy(prob_map.cpu().numpy() > self.prior)
    
    def get_probability_map_entropy(self, e) -> float:
        """Get local probability map entropy for an environment."""
        prob_map = self.get_probability_map_normalized(e)
        return np.sum(-prob_map * np.log(prob_map + 1e-5)).item()
    
    def get_local_voxel(self,e) -> torch.tensor:
        """Get local voxel map for an environment."""
        voxel = self.local_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :].permute(2,1,0).cpu() # [H, W, C]
        return voxel
    
    def get_global_voxel(self,e) -> torch.tensor:
        """Get global voxel map for an environment."""
        return self.global_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :].clone() # [H, M, M]

    def get_local_pointcloud(self, e,transform_to_hab_world_frame=True, to_meter=True) -> torch.tensor:
        """Get cropped local point cloud for an environment."""
        # TODO: currently not current
        voxel = self.local_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :]

        
        if transform_to_hab_world_frame:
            voxel = voxel.permute(2,1,0)
            pc = torch.nonzero(~voxel.isinf()).float() # [N, 3]
            pc[:,:2] += self.origins[e, 0:2:-1] # [N, 3], transform to global map frame
        else:
            voxel = voxel.permute(2,0,1)
            pc = torch.nonzero(~voxel.isinf()).float() # [N, 3]
            pc[:,:2] += self.origins[e, :2] # [N, 3], transform to global map frame
        
        pc[:,:2] -= self.global_map_size / 2
        pc = pc * self.resolution + self.resolution / 2 # [N, 3] 
        
        if to_meter:
            return pc / 100
        return pc
    
    def get_global_pointcloud(self, e ) -> torch.tensor:
        """Get global point cloud for an environment.
        returns:
            pc: tensor of shape [N, 3] in habitat world frame (x: forward, y: left, z: up), unit: meter
            feat: tensor of shape [N, 1], prob of occupancy [0, 1]
           
        """
        voxel = self.global_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :]
        voxel = voxel.permute(2,1,0) # in hab world frame (x: forward, y: left, z: up)
        idx = torch.nonzero(~voxel.isinf()) # [N, 3]
        feat = voxel[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(1) # [N, 1]
        pc = idx.float() # [N, 3]
        pc[:,:2] -= self.global_map_size / 2 
        pc = pc * self.resolution  # [N, 3] 
        
        feat = torch.sigmoid(feat)
        
        return pc / 100, feat

    def get_global_pointcloud_flat_idx(self, e ) -> torch.tensor:
        """
        returns: tensor if shape [N, 1] of flat indices of global point cloud, type is long
        """
        voxel = self.global_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :]
        voxel = voxel.permute(2,1,0) # in hab world frame (x: forward, y: left, z: up)
        idx = torch.nonzero(~voxel.isinf()) # [N, 3]
        return idx[:,0] * self.global_map_size * self.global_map_size + idx[:,1] * self.global_map_size + idx[:,2]
        
    def get_num_points(self, e) -> int:
        """Get number of points in global point cloud for an environment."""
        voxel = self.global_map[e, MC.VOXEL_START:MC.NON_SEM_CHANNELS, :, :]
        return torch.sum(~voxel.isinf()).item()
    
    def get_num_explored_cells(self, e) -> int:
        """Get number of explored cells in global map for an environment."""
        return torch.sum(self.global_map[e, MC.EXPLORED_MAP, :, :]).item()
    
    def get_num_promising_cells(self, e) -> int:
        """Get number of promising cells in global map for an environment."""
        return torch.sum(self.global_map[e, MC.PROBABILITY_MAP, :, :] > self.prior_logit).item()
    
    def get_dialated_obstacle_map_local(self, e, size=0) -> np.ndarray:
        """Get dialated local obstacle map for an environment."""
        obstacle_map = self.local_map[e, MC.OBSTACLE_MAP, :, :]
        if size > 0:
            obstacle_map = dialate_tensor(obstacle_map.unsqueeze(0), size)[0]
        obstacle_map = obstacle_map > 0
        return np.copy(obstacle_map.cpu().float().numpy())

    def get_dialated_obstacle_map_global(self, e, size=3) -> np.ndarray:
        """Get dialated local obstacle map for an environment."""
        obstacle_map = self.global_map[e, MC.OBSTACLE_MAP, :, :].clone()
        obstacle_map = dialate_tensor(obstacle_map.unsqueeze(0), size)[0]
        obstacle_map = obstacle_map > 0
        return obstacle_map
        
    
    def get_explored_locs(self, e) -> np.ndarray:
        """Get global explored locations for an environment.
        Returns:
            exp_locs: tensor of shape [N, 2] in global hab frame (x: forward, y: left), unit: meter
            note this frame is defined by the agenet's starting position (origin is at the center of the map)
        """
        exp_map = self.global_map[e, MC.EXPLORED_MAP].clone()

        obstacle_map = self.global_map[e, MC.OBSTACLE_MAP] # [H, W]
        obstacle_map = dialate_tensor(obstacle_map.unsqueeze(0), 5)[0] # [H, W]
        exp_map[obstacle_map > 0] = 0 # remove explored locations that are obstacles

        exp_locs = torch.nonzero(exp_map).float() # [N, 2] of (x, y)

        exp_locs = exp_locs * self.resolution 
        exp_locs = exp_locs[:, [1, 0]] / 100 # local hab frame: x: forward, y: left

        exp_locs -= self.global_map_size_cm / 2 / 100 # local hab frame: x: forward, y: left        
        return exp_locs
        
    def hab_world_to_map_global_frame(self, e, pos):
        """
        Args:
            pos: tensor of shape [N, 2] in habitat world frame (x: forward, y: left), unit: meter
            note this frame is defined by the agenet's starting position (origin is at the center of the map)

        Returns:
            pos: tensor of shape [N, 2] in map frame (x: right, y: down)
        """
        pos_map = pos + self.global_map_size_cm / 2 / 100 # first transform to the eps global hab frame 

        pos_map = pos_map * 100
        pos_map = pos_map // self.resolution # in grids
        pos_map = pos_map[:, [1, 0]] # in grids, global map frame: x: right, y: down, with origin at the center

        # pos_map = pos_map + self.global_map_size / 2 # in grids, global map frame: x: right, y: down, with origin at the top left
        return pos_map.long()

    def hab_world_to_map_local_frame(self, e, pos):
        """
        Args:
            pos: tensor of shape [N, 2] in habitat world frame (x: forward, y: left), unit: meter
            note this frame is defined by the agenet's starting position (origin is at the center of the map)
        Returns:
            pos: tensor of shape [N, 2] in map frame (x: right, y: down)
        """
        pos_map = pos + self.global_map_size_cm / 2 / 100 # first transform to the eps global hab frame 

        pos_map = pos_map - self.origins[e, :2] # [N, 2], transform to local hab frame

        pos_map = pos_map * 100
        pos_map = pos_map // self.resolution # in grids, local hab frame: x: forward, y: left
        pos_map = pos_map[:, [1, 0]] # in grids, local map frame: x: right, y: down

        # we clamp the position to be within the local map
        # TODO: this is not ideal, we should instead set the pos to the closest point
        #  that is navigable and closest to the original pos
        pos_map = torch.clamp(pos_map, 
                              min=0, 
                              max=self.local_map_size - 1) # in grids, local map frame: x: right, y: down
        return pos_map.long()
    
    def local_map_coords_to_hab_world_frame(self, e, pos):
        """
        Args:
            pos: tensor of shape [N, 2] in local map frame (x: right, y: down), 
        Returns:
            pos: tensor of shape [N, 2] in habitat world frame (x: forward, y: left), unit: meter
        """
        pos = pos[:, [1, 0]] * self.resolution / 100 
        pos = pos + self.origins[e, :2] # [N, 2], transform to global hab frame
        pos = pos - self.global_map_size_cm / 2 / 100 # [N, 2], transform to global hab frame
        return pos
    
    def get_exp_coverage_area(self, e) -> float:
        """Get the total area of the map that is explored."""
        exp_area = (self.global_map[e, MC.EXPLORED_MAP, :, :] > 0).sum().item()
        return exp_area * self.resolution * self.resolution / 10000 # in m^2
    
    def get_close_coverage_area(self, e) -> float:
        """Get the total area of the map that is explored."""
        close_area = (self.global_map[e, MC.BEEN_CLOSE_MAP, :, :] > 0).sum().item()
        return close_area * self.resolution * self.resolution / 10000 # in m^2
    
    def get_planner_pose_inputs(self, e) -> np.ndarray:
        """Get local planner pose inputs for an environment.

        Returns:
            planner_pose_inputs with 7 dimensions:
             1-3: Global pose
             4-7: Local map boundaries
        """
        return (
            torch.cat([self.local_pose[e] + self.origins[e], self.lmb[e]])
            .cpu()
            .float()
            .numpy()
        )

    def get_goal_map(self, e) -> np.ndarray:
        """Get binary goal map encoding current global goal for an
        environment."""
        return self.goal_map[e]
