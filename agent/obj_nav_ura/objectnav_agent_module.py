# Adapted from https://github.com/facebookresearch/home-robot

import time

import torch.nn as nn

from mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)

from .ur_policy import ObjectNavFrontierExplorationPolicy

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points
from utils.geometry.urp_dr import DRPlanner
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
import trimesh.transformations as tra



# Do we need to visualize the frontier as we explore?
debug_frontier_map = False


class ObjectNavAgentModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.ENVIRONMENT.frame_height,
            frame_width=config.ENVIRONMENT.frame_width,
            camera_height=config.ENVIRONMENT.camera_height,
            hfov=config.ENVIRONMENT.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            min_depth=config.ENVIRONMENT.min_depth,
            max_depth=config.ENVIRONMENT.max_depth,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
            probabilistic=config.AGENT.SEMANTIC_MAP.use_probability_map,
            probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior,
            close_range=config.AGENT.SEMANTIC_MAP.close_range,
            confident_threshold=config.AGENT.SEMANTIC_MAP.confident_threshold,
        )
        self.policy = ObjectNavFrontierExplorationPolicy(
            exploration_strategy=config.AGENT.exploration_strategy
        )
 

    @property
    def goal_update_steps(self):
        return self.policy.goal_update_steps
    
    # def init_dr_planner(self,obs):
    #     """
    #     we need obs to init the dr planner
    #     """
    #     if 'camera_pose' in obs.__dict__:
    #         angles = tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")
    #         self.camera_tilt = - angles[1]

    #         # Get the camera height
    #         self.camera_height = obs.camera_pose[2, 3]
    #     else:
    #         # otherwise use the default values
    #         self.camera_tilt = 0
    #         self.camera_height = self.config.ENVIRONMENT.camera_height

    #     # currently only support a single environment
    #     self.dr_planner = DRPlanner(self.camera_tilt, self.camera_height, self.config, self.device)

    def forward(
        self,
        seq_obs,
        seq_pose_delta,
        seq_dones,
        seq_update_global,
        seq_camera_poses,
        init_local_map,
        init_global_map,
        init_local_pose,
        init_global_pose,
        init_lmb,
        init_origins,
        seq_object_goal_category=None,
        seq_start_recep_goal_category=None,
        seq_end_recep_goal_category=None,
        seq_nav_to_recep=None,
        detection_results=None,
    ):
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)
            seq_object_goal_category: sequence of object goal categories of shape
             (batch_size, sequence_length, 1)
            seq_start_recep_goal_category: sequence of start recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_end_recep_goal_category: sequence of end recep goal categories of shape
             (batch_size, sequence_length, 1)
            seq_nav_to_recep: sequence of binary digits indicating if navigation is to object or end receptacle of shape
             (batch_size, 1)
        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # t0 = time.time()

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            seq_extras,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
            detection_results
        )

        # t1 = time.time()
        # print(f"[Semantic mapping] Total time: {t1 - t0:.2f}")

        # Predict high-level goals from map features
        # batched across sequence length x num environments
        map_features = seq_map_features.flatten(0, 1)
        if seq_object_goal_category is not None:
            seq_object_goal_category = seq_object_goal_category.flatten(0, 1)
        if seq_start_recep_goal_category is not None:
            seq_start_recep_goal_category = seq_start_recep_goal_category.flatten(0, 1)
        if seq_end_recep_goal_category is not None:
            seq_end_recep_goal_category = seq_end_recep_goal_category.flatten(0, 1)
        
        # Compute the goal map
        goal_map, found_goal = self.policy(
            map_features,
            seq_object_goal_category,
            seq_start_recep_goal_category,
            seq_end_recep_goal_category,
            seq_nav_to_recep,
        )
        seq_goal_map = goal_map.view(batch_size, sequence_length, *goal_map.shape[-2:])
        seq_found_goal = found_goal.view(batch_size, sequence_length)

        # Compute the frontier map here
        frontier_map = self.policy.get_frontier_map(map_features)
        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )
        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(seq_frontier_map[0, 0].numpy())
            plt.subplot(122)
            plt.imshow(goal_map[0].numpy())
            plt.show()
            breakpoint()
        # t2 = time.time()
        # print(f"[Policy] Total time: {t2 - t1:.2f}")

        ################# UR policy ################








        ###############################################
        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            seq_extras,
        )

    # def _get_dense_info_gain_map(self, semantic_map, e):
    #     """
    #     return the dense info gain map
        
    #     """
    #     exp_pos_all = semantic_map.get_explored_locs(e) # [N, 2]
    #     n_p = exp_pos_all.shape[0]
    #     print(f'Num of points: {n_p}')
    #     if n_p > self._max_render_loc:
    #         if self._uniform_sampling:
    #             rng = np.random.default_rng()
    #             arr = np.arange(n_p)
    #             rng.shuffle(arr)
    #             exp_pos = exp_pos_all[arr[:self._max_render_loc]]

    #         else:
    #             exp_pos = sample_farthest_points(exp_pos_all.unsqueeze(0), K=self._max_render_loc)[0] # [1, N, 2]
    #             exp_pos = exp_pos.squeeze(0) # [N, 2]

    #     points, feats = semantic_map.get_global_pointcloud(e) # [P, 3], [P, 1]
    #     points = points.unsqueeze(0) # [1, P, 3]
    #     feats = feats.unsqueeze(0) # [1, P, 1]
    #     info_at_locs = self.dr_planner.cal_info_gains(points,feats,exp_pos)

    #     # visualize
    #     info_sorted, idx = torch.sort(info_at_locs, descending=True)
    #     exp_pos_map = semantic_map.hab_world_to_map_global_frame(e, exp_pos).long()
    #     exp_pos_map = exp_pos_map[idx].cpu() # sorted

    #     local_map_color = torch.zeros_like(semantic_map.global_map[e,0]) 
    #     local_map_color = local_map_color.unsqueeze(-1).repeat(1,1,3).cpu()
        
    #     # mark first 5 points as red
    #     local_map_color[exp_pos_map[:5,0], exp_pos_map[:5,1], :] = torch.tensor([1.,0.,0.])
    #     # mark the last 5 points as green
    #     local_map_color[exp_pos_map[-5:,0], exp_pos_map[-5:,1], :] = torch.tensor([0.,1.,0.])
    #     # mark the rest as grey
    #     local_map_color[exp_pos_map[5:-5,0], exp_pos_map[5:-5,1], :] = torch.tensor([0.5,0.5,0.5])
        
    #     # display all
    #     local_map = torch.zeros_like(semantic_map.global_map[e,0])
    #     local_map[exp_pos_map[:,0], exp_pos_map[:,1]] = 1.0

    #     local_map_color = local_map_color.cpu().numpy()
    #     return local_map_color