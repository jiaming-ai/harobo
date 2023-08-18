# Adapted from https://github.com/facebookresearch/home-robot


from typing import Tuple, Optional, Dict, List
from torch.nn.utils.rnn import pad_sequence
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
import torch
import torch.nn as nn
import trimesh.transformations as tra
from torch import IntTensor, Tensor
from torch.nn import functional as F

import home_robot.mapping.map_utils as mu
from utils.geometry import depth as du
from utils.geometry.points_utils import show_points, get_pc_from_voxel, show_voxel
import home_robot.utils.pose as pu
import home_robot.utils.rotation as ru
from mapping.semantic.constants import MapConstants as MC
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


# For debugging input and output maps - shows matplotlib visuals
debug_maps = False


class Categorical2DSemanticMapModule(nn.Module):
    """
    This class is responsible for updating a dense 2D semantic map with one channel
    per object category, the local and global maps and poses, and generating
    map features — it is a stateless PyTorch module with no trainable parameters.

    Map proposed in:
    Object Goal Navigation using Goal-Oriented Semantic Exploration
    https://arxiv.org/pdf/2007.00643.pdf
    https://github.com/devendrachaplot/Object-Goal-Navigation
    """

    # If true, display point cloud visualizations using Open3d
    debug_mode = False

    def __init__(
        self,
        # config,
        frame_height: int,
        frame_width: int,
        camera_height: int,
        hfov: int,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        vision_range: int,
        explored_radius: int,
        been_close_to_radius: int,
        global_downscaling: int,
        du_scale: int,
        cat_pred_threshold: float,
        exp_pred_threshold: float,
        map_pred_threshold: float,
        min_depth: float = 0.2,
        max_depth: float = 5.0,
        must_explore_close: bool = False,
        min_obs_height_cm: int = 0,
        dilate_obstacles: bool = True,
        dilate_iter: int = 1,
        dilate_size: int = 3,
        probabilistic: bool = False,
        probability_prior: float = 0.2,
        close_range: int = 150, # 1.5m
        confident_threshold: float = 0.7,
    ):
        """
        Arguments:
            frame_height: first-person frame height
            frame_width: first-person frame width
            camera_height: camera sensor height (in metres)
            hfov: horizontal field of view (in degrees)
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimetres)
            map_resolution: size of map bins (in centimeters)
            vision_range: diameter of the circular region of the local map
             that is visible by the agent located in its center (unit is
             the number of local map cells)
            explored_radius: radius (in centimeters) of region of the visual cone
             that will be marked as explored
            been_close_to_radius: radius (in centimeters) of been close to region
            global_downscaling: ratio of global over local map
            du_scale: frame downscaling before projecting to point cloud
            cat_pred_threshold: number of depth points to be in bin to
             classify it as a certain semantic category
            exp_pred_threshold: number of depth points to be in bin to
             consider it as explored
            map_pred_threshold: number of depth points to be in bin to
             consider it as obstacle
            must_explore_close: reduce the distance we need to get to things to make them work
            min_obs_height_cm: minimum height of obstacles (in centimetres)
        """
        super().__init__()

        # frame_height=config.ENVIRONMENT.frame_height,
        # frame_width=config.ENVIRONMENT.frame_width,
        # camera_height=config.ENVIRONMENT.camera_height,
        # hfov=config.ENVIRONMENT.hfov,
        # num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
        # map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
        # map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
        # vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
        # min_depth=config.ENVIRONMENT.min_depth,
        # max_depth=config.ENVIRONMENT.max_depth,
        # explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
        # been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
        # global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
        # du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
        # cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
        # exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
        # map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
        # must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
        # min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
        # dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
        # dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
        # dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
        # probabilistic=config.AGENT.SEMANTIC_MAP.use_probability_map,
        # probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior,
        # close_range=config.AGENT.SEMANTIC_MAP.close_range,
        # confident_threshold=config.AGENT.SEMANTIC_MAP.confident_threshold,

        self.screen_h = frame_height
        self.screen_w = frame_width
        self.camera_matrix = du.get_camera_matrix(self.screen_w, self.screen_h, hfov)
        self.num_sem_categories = num_sem_categories
        self.must_explore_close = must_explore_close

        self.map_size_parameters = mu.MapSizeParameters(
            map_resolution, map_size_cm, global_downscaling
        )
        self.resolution = map_resolution
        self.global_map_size_cm = map_size_cm
        self.global_downscaling = global_downscaling
        self.local_map_size_cm = self.global_map_size_cm // self.global_downscaling
        self.global_map_size = self.global_map_size_cm // self.resolution
        self.local_map_size = self.local_map_size_cm // self.resolution
        self.xy_resolution = self.z_resolution = map_resolution
        self.vision_range = vision_range
        self.explored_radius = explored_radius
        self.been_close_to_radius = been_close_to_radius
        self.du_scale = du_scale
        self.cat_pred_threshold = cat_pred_threshold
        self.exp_pred_threshold = exp_pred_threshold
        self.map_pred_threshold = map_pred_threshold

        self.max_depth = max_depth * 100.0
        self.min_depth = min_depth * 100.0
        self.agent_height = camera_height * 100.0
        self.max_voxel_height = int(360 / self.z_resolution)
        self.min_voxel_height = int(-40 / self.z_resolution)
        self.min_obs_height_cm = min_obs_height_cm
        self.min_mapped_height = int(
            self.min_obs_height_cm / self.z_resolution - self.min_voxel_height
        )
        # ignore the ground
        self.filtered_min_height = int(
            20 / self.z_resolution - self.min_voxel_height
        )  # 20cm
        self.max_mapped_height = int(
            (self.agent_height + 1) / self.z_resolution - self.min_voxel_height
        )
        self.shift_loc = [self.vision_range * self.xy_resolution // 2, 0, np.pi / 2.0]

        # For cleaning up maps
        self.dilate_obstacles = dilate_obstacles
        # self.dilate_kernel = np.ones((dilate_size, dilate_size))
        self.dilate_size = dilate_size
        # self.dilate_iter = dilate_iter
        
        self.probabilistic = probabilistic
        # For probabilistic map updates
        self.dist_rows = torch.arange(1, self.vision_range + 1).float()
        self.dist_rows = self.dist_rows.unsqueeze(1).repeat(1, self.vision_range)
        self.dist_cols = torch.arange(1, self.vision_range + 1).float() - (self.vision_range / 2)
        self.dist_cols = torch.abs(self.dist_cols)
        self.dist_cols = self.dist_cols.unsqueeze(0).repeat(self.vision_range, 1)

        self.close_range = close_range // self.xy_resolution # 150 cm
        self.confident_threshold = confident_threshold # above which considered a hard detection
        self.prior_logit = torch.logit(torch.tensor(probability_prior)) # prior probability of objects
        self.vr_matrix = torch.zeros((1, self.vision_range, self.vision_range))
        self.prior_matrix = torch.full((1, self.vision_range, self.vision_range), self.prior_logit)

        self.dialate_kernel = torch.ones((1, 1, dilate_size, dilate_size), dtype=torch.float32)

    @torch.no_grad()
    def forward(
        self,
        seq_obs: Tensor,
        seq_pose_delta: Tensor,
        seq_dones: Tensor,
        seq_update_global: Tensor,
        seq_camera_poses: Tensor,
        init_local_map: Tensor,
        init_global_map: Tensor,
        init_local_pose: Tensor,
        init_global_pose: Tensor,
        init_lmb: Tensor,
        init_origins: Tensor,
        detection_results: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, IntTensor, Tensor]:
        """Update maps and poses with a sequence of observations and generate map
        features at each time step.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) binary flags
             that indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, 4, 4) extrinsic camera
             matrices
            init_local_map: initial local map before any updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_map_features: sequence of semantic map features of shape
             (batch_size, sequence_length, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_local_map: final local map after all updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
            detection_results: list of detection results of length sequence_length
        """
        batch_size, sequence_length = seq_obs.shape[:2]
        device, dtype = seq_obs.device, seq_obs.dtype
        detection_results = detection_results or [None] * sequence_length
        
        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories
        seq_map_features = torch.zeros(
            batch_size,
            sequence_length,
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=device,
            dtype=dtype,
        )
        # n_points = self.screen_h // self.du_scale * self.screen_w // self.du_scale
        # # n_points = self.screen_h  * self.screen_w 
        # seq_point_clouds = torch.zeros(
        #     batch_size,
        #     sequence_length,
        #     n_points,
        #     3,
        #     device=device,
        #     dtype=dtype,
        # )
        seq_local_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_global_pose = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_lmb = torch.zeros(
            batch_size, sequence_length, 4, device=device, dtype=torch.int32
        )
        seq_origins = torch.zeros(batch_size, sequence_length, 3, device=device)
        seq_extras = [None] * sequence_length

        local_map, local_pose = init_local_map.clone(), init_local_pose.clone()
        global_map, global_pose = init_global_map.clone(), init_global_pose.clone()
        lmb, origins = init_lmb.clone(), init_origins.clone()
        for t in range(sequence_length):
            # Reset map and pose for episodes done at time step t
            for e in range(batch_size):
                if seq_dones[e, t]:
                    mu.init_map_and_pose_for_env(
                        e,
                        local_map,
                        global_map,
                        local_pose,
                        global_pose,
                        lmb,
                        origins,
                        self.map_size_parameters,
                    )

            local_map, local_pose, extras = self._update_local_map_and_pose(
                seq_obs[:, t],
                seq_pose_delta[:, t],
                local_map,
                local_pose,
                seq_camera_poses,
                detection_results[t],
            )
            for e in range(batch_size):
                if seq_update_global[e, t]:
                    self._update_global_map_and_pose_for_env(
                        e, local_map, global_map, local_pose, global_pose, lmb, origins
                    )

            seq_local_pose[:, t] = local_pose
            seq_global_pose[:, t] = global_pose
            seq_lmb[:, t] = lmb
            seq_origins[:, t] = origins
            seq_map_features[:, t] = self._get_map_features(local_map, global_map)
            seq_extras[t] = extras

        return (
            seq_map_features,
            local_map,
            global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            seq_extras,
        )

    def _update_local_map_and_pose(
        self,
        obs: Tensor,
        pose_delta: Tensor,
        prev_map: Tensor,
        prev_pose: Tensor,
        camera_pose: Tensor,
        detection_result: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Update local map and sensor pose given a new observation using parameter-free
        differentiable projective geometry.

        Args:
            obs: current frame containing (rgb, depth, segmentation) of shape
             (batch_size, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: delta in pose since last frame of shape (batch_size, 3)
            prev_map: previous local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            prev_pose: previous pose of shape (batch_size, 3)
            camera_pose: current camera poseof shape (batch_size, 4, 4)

        Returns:
            current_map: current local map updated with current observation
             and location of shape (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            current_pose: current pose updated with pose delta of shape (batch_size, 3)
        """
        batch_size, obs_channels, h, w = obs.size()
        device, dtype = obs.device, obs.dtype
        if camera_pose is not None:
            # TODO: make consistent between sim and real
            # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
            # angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
            angles = torch.Tensor(
                [tra.euler_from_matrix(p[:3, :3].cpu(), "rzyx") for p in camera_pose]
            )
            # For habitat - pull x angle
            # tilt = angles[:, -1]
            # For real robot
            tilt = angles[:, 1]

            # Get the agent pose
            # hab_agent_height = camera_pose[:, 1, 3] * 100
            agent_pos = camera_pose[:, :3, 3] * 100
            agent_height = agent_pos[:, 2]
        else:
            tilt = torch.zeros(batch_size)
            agent_height = self.agent_height

        depth = obs[:, 3, :, :].float()
        depth[depth > self.max_depth] = 0
        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, device, scale=self.du_scale
        )

        if self.debug_mode:
            from home_robot.utils.point_cloud import show_point_cloud

            rgb = obs[:, :3, :: self.du_scale, :: self.du_scale].permute(0, 2, 3, 1)
            xyz = point_cloud_t[0].reshape(-1, 3)
            rgb = rgb[0].reshape(-1, 3)
            print("-> Showing point cloud in camera coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        point_cloud_base_coords = du.transform_camera_view_t(
            point_cloud_t, agent_height, torch.rad2deg(tilt).cpu().numpy(), device
        )

        # Show the point cloud in base coordinates for debugging
        if self.debug_mode:
            print()
            print("------------------------------")
            print("agent angles =", angles)
            print("agent tilt   =", tilt)
            print("agent height =", agent_height, "preset =", self.agent_height)
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in base coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        #################### pointclouds ####################
        
        # xyz = point_cloud_base_coords.clone().reshape(batch_size,-1, 3)

        # we use the detection score as feat for point cloud
        # we assume total_num_instance is the same for all batch (padded with 0)

        # first take max prob value for pixel
        prob_feat = torch.zeros(batch_size, 1, h // self.du_scale * w // self.du_scale).to(device)
        if detection_result is not None:
            scores = detection_result["scores"] # [B, total_num_instance]
            classes = detection_result["classes"] # [B, total_num_instance]
            masks = detection_result["masks"].float() # [B, total_num_instance, H, W]
            relevance = detection_result["relevance"] #torch.tensor([0, 1, 0.7 , 0, 0]).to(device)

            if masks.shape[1] != 0: # no instance detected
                score_relevence = scores * relevance[classes] # [B, total_num_instance]
                prob_feat = torch.einsum('bnhw,bn->bnhw',masks, score_relevence) # [B, N, H, W]
                prob_feat,_ = torch.max(prob_feat, dim=1, keepdim=True) # [B,1, H, W]
                # we use maxpool2d instead of avgpool2d to preserve the prob value
                prob_feat = nn.MaxPool2d(self.du_scale)(prob_feat).view(
                        batch_size, 1,  h // self.du_scale * w // self.du_scale
                    ) # [B, 1,  H*W] after scaling
        #################### pointclouds ####################

        point_cloud_map_coords = du.transform_pose_t(
            point_cloud_base_coords, self.shift_loc, device
        )

        if self.debug_mode:
            xyz = point_cloud_base_coords[0].reshape(-1, 3)
            print("-> Showing point cloud in map coords")
            show_point_cloud(
                (xyz / 100.0).cpu().numpy(),
                (rgb / 255.0).cpu().numpy(),
                orig=np.zeros(3),
            )

        voxel_channels = 2 + self.num_sem_categories # first is for 3d structure, last is for prob feat
        
        init_grid = torch.zeros(
            batch_size,
            voxel_channels,
            self.vision_range,
            self.vision_range,
            self.max_voxel_height - self.min_voxel_height,
            device=device,
            dtype=torch.float32,
        )
        feat = torch.ones(
            batch_size,
            voxel_channels-1, # cat + prob
            self.screen_h // self.du_scale * self.screen_w // self.du_scale,
            device=device,
            dtype=torch.float32,
        )

        # PMO
        # if num_instance > 0:
        #     feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(masks).view(
        #         batch_size, num_instance, h // self.du_scale * w // self.du_scale
        #     )

        # feat: 0 is for explored area, 1:-1 is for instance, -1 is for prob
        feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(obs[:, 4:, :, :]).view(
            batch_size, obs_channels - 4, h // self.du_scale * w // self.du_scale
        )
        feat = torch.cat([feat,prob_feat],dim=1)

        XYZ_cm_std = point_cloud_map_coords.float()
        XYZ_cm_std[..., :2] = XYZ_cm_std[..., :2] / self.xy_resolution
        XYZ_cm_std[..., :2] = (
            (XYZ_cm_std[..., :2] - self.vision_range // 2.0) / self.vision_range * 2.0
        )
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / self.z_resolution
        XYZ_cm_std[..., 2] = (
            (
                XYZ_cm_std[..., 2]
                - (self.max_voxel_height + self.min_voxel_height) // 2.0
            )
            / (self.max_voxel_height - self.min_voxel_height)
            * 2.0
        )
        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2) # [B, 3, H, W]
        XYZ_cm_std = XYZ_cm_std.view(
            XYZ_cm_std.shape[0],
            XYZ_cm_std.shape[1],
            XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3],
        ) # [B, 3, H*W]

        # voxels = du.splat_feat_nd_max(init_grid, feat, XYZ_cm_std).transpose(2, 3)
        voxels = du.splat_feat_nd(init_grid, feat, XYZ_cm_std).transpose(2, 3)

        all_height_proj = voxels[:,:1,...].sum(4)
        # ignore objects that are too low
        filtered_height_proj = voxels[...,
            self.filtered_min_height : self.max_mapped_height
        ].sum(4)
        # the agent_height range corresponds to 0cm to 120cm 
        agent_height_proj = voxels[:,:1,:,:,
            self.min_mapped_height : self.max_mapped_height
        ].sum(4)
     
        fp_map_pred = agent_height_proj[:, 0:1, :, :]
        fp_exp_pred = all_height_proj[:, 0:1, :, :]
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold

        close_exp = fp_exp_pred.clone() # [B, 1, H, W]
        close_exp[:,:, self.close_range:,:] = 0 # only consider the close range 1.5m
        
        ################ probabilitic ################
        
        ############ PMO ##############
        # # TODO filter out voxels that are lower than the threshold
        # # this is based on the assumption that the object should not be on the ground
        # instance_projection = voxels[:, 1:, :, :,
        #      self.min_mapped_height + 4 : self.max_mapped_height
        # ].sum(4) # [B, total_num_instance, H, W]

        # ### We need to pad the instances to the same size
        # probs = torch.zeros([batch_size, 
        #                         self.num_sem_categories,
        #                         self.vision_range,
        #                         self.vision_range], 
        #                         dtype=torch.float32,
        #                         device=device)
        
        # for c in [1,2,3]: # obj, rec, goal_rec
        #     idx_b, idx_c = torch.where(classes==c) 
        #     if len(idx_b) == 0: # no detection for all batches
        #         continue

        #     # combine the instances from the same class by taking the max
        #     instances_list = instance_projection[[idx_b, idx_c]].clamp(0,1) # [c_total_num_instance, H, W]
        #     instances_list *= scores[[idx_b, idx_c]].unsqueeze(-1).unsqueeze(-1) # [c_total_num_instance, H, W]
        #     idx_b_list = idx_b.tolist()
        #     part_sizes = [idx_b_list.count(i) for i in range(batch_size)]

        #     instances_list = torch.split(instances_list, part_sizes, dim=0)
        #     instances_tensor = pad_sequence(instances_list, batch_first=True) # [B, c_num_instance, H, W]
        #     combined, _ = torch.max(instances_tensor, dim=1)   # [B, H, W]

        #     probs[:, c, :, :] = combined

        # # assign hard detection results
        # detected = probs > self.confident_threshold

        # # relevance = torch.tensor([0, 1, 0.7 , 0, 0]).to(device)
        # prob_map = torch.einsum('bchw,c->bhw',probs,relevance) # [B, H, W]

        # # we only reduce the probablities for the close range that is viewable by the agent
        # # prior_matrix = self.prior_matrix.repeat(batch_size, 1, 1) # [B, H, W]
        # # prior_matrix[fp_exp_pred.squeeze(1) > 0] = self.prior_logit # [B, H, W]

        ########### end PMO ###################
        prob_map, _ = voxels[:,-1,:,:,self.filtered_min_height : self.max_mapped_height].max(3)

        # TODO: should we use close_range or exp, or just all viewable area?
        # we can use a smaller prior for all viewable area, and bigger prior for close range
        prob_logit = torch.logit(prob_map,eps=1e-6) - self.prior_logit # 
        prob_logit[fp_exp_pred.squeeze(1) == 0] = 0 # set unviewable area to 0
        prob_logit = torch.clamp(prob_logit, min=-10, max=10)        

        

        ############### end probabilistic ###############


        agent_view = torch.zeros(
            batch_size,
            MC.NON_SEM_CHANNELS + self.num_sem_categories,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )

        # Update agent view from the fp_map_pred
        if self.dilate_obstacles:
            
            fp_map_pred =  torch.nn.functional.conv2d(
                fp_map_pred, self.dialate_kernel.to(device), padding=self.dilate_size // 2
            ).clamp(0, 1)
            # for i in range(fp_map_pred.shape[0]):
            #     env_map = fp_map_pred[i, 0].cpu().numpy()
            #     # TODO: remove if not used
            #     # env_map_eroded = cv2.erode(
            #     #     env_map, self.dilate_kernel, self.dilate_iter
            #     # )
            #     # filt = cv2.filter2D(env_map, -1, self.dilate_kernel)
            #     median_filtered = cv2.medianBlur(env_map, self.dilate_size)

            #     # TODO: remove debug code
            #     # plt.subplot(121); plt.imshow(env_map)
            #     # plt.subplot(122); plt.imshow(env_map_eroded)
            #     # plt.show()
            #     # breakpoint()
            #     # fp_map_pred[i, 0] = torch.tensor(env_map_eroded)
            #     fp_map_pred[i, 0] = torch.tensor(median_filtered)

        x1 = self.local_map_size_cm // (self.xy_resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.local_map_size_cm // (self.xy_resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, MC.OBSTACLE_MAP : MC.OBSTACLE_MAP + 1, y1:y2, x1:x2] = fp_map_pred
        agent_view[:, MC.EXPLORED_MAP : MC.EXPLORED_MAP + 1, y1:y2, x1:x2] = fp_exp_pred
        agent_view[:, MC.BEEN_CLOSE_MAP : MC.BEEN_CLOSE_MAP + 1, y1:y2, x1:x2] = close_exp
        agent_view[:, MC.PROBABILITY_MAP , y1:y2, x1:x2] = prob_logit
        agent_view[:, MC.VOXEL_START: MC.NON_SEM_CHANNELS, y1:y2, x1:x2] = voxels[:,-1,:,:,
            : self.max_mapped_height
        ].permute(0,3,1,2) # [B, H, W, C] -> [B, C, H, W]
        
        # PMO
        # agent_view[
        #     :,
        #     MC.NON_SEM_CHANNELS : MC.NON_SEM_CHANNELS + self.num_sem_categories,
        #     y1:y2,
        #     x1:x2,
        # ] = (
        #     detected
        #     / self.cat_pred_threshold
        # )
        agent_view[:, MC.NON_SEM_CHANNELS :, y1:y2, x1:x2] = (
            filtered_height_proj[:, 1:-1] / self.cat_pred_threshold
        )
        
        #### for voxel ####
        occupaid_voxel = torch.zeros(
            batch_size,
            self.max_mapped_height,
            self.local_map_size_cm // self.xy_resolution,
            self.local_map_size_cm // self.xy_resolution,
            device=device,
            dtype=dtype,
        )
            
        occupaid_voxel[..., y1:y2, x1:x2] = voxels[:,0,:,:, : self.max_mapped_height].permute(0,3,1,2)
        agent_view = torch.cat([agent_view, occupaid_voxel], dim=1)
        ####################
        
        current_pose = pu.get_new_pose_batch(prev_pose.clone(), pose_delta)
        st_pose = current_pose.clone().detach()

        st_pose[:, :2] = -(
            (
                st_pose[:, :2] * 100.0 / self.xy_resolution
                - self.local_map_size_cm // (self.xy_resolution * 2)
            )
            / (self.local_map_size_cm // (self.xy_resolution * 2))
        )
        st_pose[:, 2] = 90.0 - (st_pose[:, 2])

        rot_mat, trans_mat = ru.get_grid(st_pose, agent_view.size(), dtype)
        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        
        #### for voxel ####
        occupaid_voxel_st = translated[:,MC.NON_SEM_CHANNELS+self.num_sem_categories:,:,:]
        translated = translated[:,:MC.NON_SEM_CHANNELS+self.num_sem_categories,:,:]
        ####################

        # Clamp to [0, 1] after transform agent view to map coordinates
        idx_no_prob = list(range(0,MC.PROBABILITY_MAP)) + \
                    list(range(MC.NON_SEM_CHANNELS, MC.NON_SEM_CHANNELS+self.num_sem_categories))
        translated[:,idx_no_prob] = torch.clamp(translated[:,idx_no_prob], min=0.0, max=1.0)

        maps = torch.cat((prev_map.unsqueeze(1), translated.unsqueeze(1)), 1)
        current_map, _ = torch.max(
            maps[:, :, : MC.NON_SEM_CHANNELS + self.num_sem_categories], 1
        )

        ############### Bayesian update ###############
        current_map[:, MC.PROBABILITY_MAP, :, :] = torch.sum(maps[:, :, MC.PROBABILITY_MAP,:, :], 1)
        current_map[:, MC.PROBABILITY_MAP, :, :] = torch.clamp(current_map[:, MC.PROBABILITY_MAP, :, :], min=-10, max=10)
        goal_idx = MC.NON_SEM_CHANNELS + 1 # goal object

        # NOTE: here we mark all areas that have been close to the goal object as confident
        # we assume if we have been close to the object, we can identify the object, and update the semantic correctly
        # If we don't go to goal, then most likely the object is not the goal object on the specified receptacle
        
        # only for evaluation
        checking_area = (translated[:, MC.BEEN_CLOSE_MAP] == 1) \
                    & (current_map[:, MC.PROBABILITY_MAP] > self.prior_logit) \
                    & (prev_map[:, MC.BEEN_CLOSE_MAP] != 1) # only check the area that we have not been close to, to avoid repeated checking
        checking_area = checking_area.sum(dim=(1,2)).cpu().numpy() * self.resolution * self.resolution / 10000 # m^2
        extras = {
            "checking_area": checking_area, # ndarray of size [B]
        }
        # if been close to and the prob is very low
        # NOTE: however, this doesn't work well, because the prob can be hight for a recepticle
        # and make the agent repeatively checking the recepticle. We need to have different prob 
        # for different objects in order to make this work
        # So instead, we mark all close area to be low prob. Effectively, we only ask the agent to check
        # the area that has not been close to once, and then we can be confident about the object not being there
        # if the object detection model doesn't detect it
        # confident_no_obj = (current_map[:, MC.BEEN_CLOSE_MAP] == 1) \
        #                 & (current_map[:, MC.PROBABILITY_MAP] < self.prior_logit) 
        
        confident_no_obj = current_map[:, MC.BEEN_CLOSE_MAP] == 1

        current_map[:, MC.PROBABILITY_MAP][confident_no_obj] = -10 

        ############### end Bayesian update ###############
        
        ############## voxel update ################
        # we only update occupaid voxel (by current observation)
        # if voxel is empty in the previous map (isinf), then we assign the logit of the voxel: 
        # otherwise, update with l(p^t) = l(p^t-1) + l(p^t) - l(p)
        is_occupaid = occupaid_voxel_st >0.5
        is_pre_empty = prev_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:].isinf()
        need_assign_logit = is_occupaid & is_pre_empty
        need_addition_logit = is_occupaid & ~is_pre_empty
        voxel_logit = torch.logit(
            translated[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:],eps=1e-6)
        
        updated = prev_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:].clone()
        # case 1
        updated[need_assign_logit] = voxel_logit[need_assign_logit]
        # current_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:][need_assign_logit] = voxel_logit[need_assign_logit]
        # case 2
        updated[need_addition_logit] = prev_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:][need_addition_logit] + \
            voxel_logit[need_addition_logit] - self.prior_logit
        # current_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:][need_addition_logit] += \
        #     (voxel_logit[need_addition_logit] - self.prior_logit)
        
        is_post_occupaid = ~updated.isinf()
        updated[is_post_occupaid] = torch.clamp(updated[is_post_occupaid], min=-10, max=10)
        # is_post_occupaid = ~current_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:].isinf()
        # current_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:].clamp_(min=-10,max=10)

        
        # if the prob of a voxel is very low and is closely checked, then we set it to -10
        # NOTE: same reasoning as above.
        # However, as we don't know if the voxel is visible or not, we mark all voxels that
        # are both close and occupaid as low prob
        # we need to use an additional channel in order to know if the voxel has been 
        # closely looked at
        # confident_no_obj = ( updated < self.prior_logit ) & is_occupaid \
        #                 & (current_map[:, MC.BEEN_CLOSE_MAP].unsqueeze(1).repeat(1,self.max_mapped_height, 1,1)==1) # [B, H, W] -> [B, C, H, W
        confident_no_obj = confident_no_obj.unsqueeze(1).repeat(1,self.max_mapped_height, 1,1) # [B, H, W] -> [B, C, H, W]

        updated[confident_no_obj & is_post_occupaid] = -10
        current_map[:,MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:] = updated
        
        ############ how about use voxel map for prob map
        # is_post_occupaid_proj = is_post_occupaid.max(dim=1)[0]
        # current_map[:, MC.PROBABILITY_MAP, :, :][is_post_occupaid_proj] = \
        #     torch.max(current_map[:, MC.VOXEL_START:MC.NON_SEM_CHANNELS,:,:], dim=1)[0][is_post_occupaid_proj]
        # current_map[:, goal_idx :, :] = current_map[:,MC.PROBABILITY_MAP, :, :] > 0
        

        ############### end voxel update ###############
        # Reset current location
        # TODO: it is always in the center, do we need it?
        current_map[:, MC.CURRENT_LOCATION, :, :].fill_(0.0)
        curr_loc = current_pose[:, :2] # 245, 240 
        curr_loc = (curr_loc * 100.0 / self.xy_resolution).int()

        for e in range(batch_size):
            x, y = curr_loc[e]
            current_map[
                e,
                MC.CURRENT_LOCATION : MC.CURRENT_LOCATION + 2,
                y - 2 : y + 3,
                x - 2 : x + 3,
            ].fill_(1.0)

            # Set a disk around the agent to explored
            # This is around the current agent - we just sort of assume we know where we are
            # TODO being close to should be in the agent's camera frustum and within a certain distance
            # try:
            #     radius = 10
            #     explored_disk = torch.from_numpy(skimage.morphology.disk(radius))
            #     current_map[
            #         e,
            #         MC.EXPLORED_MAP,
            #         y - radius : y + radius + 1,
            #         x - radius : x + radius + 1,
            #     ][explored_disk == 1] = 1
            #     # Record the region the agent has been close to using a disc centered at the agent
            #     radius = self.been_close_to_radius // self.resolution
            #     been_close_disk = torch.from_numpy(skimage.morphology.disk(radius))
            #     current_map[
            #         e,
            #         MC.BEEN_CLOSE_MAP,
            #         y - radius : y + radius + 1,
            #         x - radius : x + radius + 1,
            #     ][been_close_disk == 1] = 1
            # except IndexError:
            #     pass

        if debug_maps:
            current_map = current_map.cpu()
            explored = current_map[0, MC.EXPLORED_MAP].numpy()
            been_close = current_map[0, MC.BEEN_CLOSE_MAP].numpy()
            obstacles = current_map[0, MC.OBSTACLE_MAP].numpy()
            plt.subplot(331)
            plt.axis("off")
            plt.title("explored")
            plt.imshow(explored)
            plt.subplot(332)
            plt.axis("off")
            plt.title("been close")
            plt.imshow(been_close)
            plt.subplot(233)
            plt.axis("off")
            plt.imshow(been_close * explored)
            plt.subplot(334)
            plt.axis("off")
            plt.title("obstacles")
            plt.imshow(obstacles)
            plt.subplot(335)
            plt.axis("off")
            plt.title("obstacles_eroded")

            obs_eroded = cv2.erode(obstacles, np.ones((5, 5)), iterations=5)
            plt.imshow(obs_eroded)
            plt.subplot(336)
            plt.axis("off")
            plt.imshow(been_close * obstacles)
            plt.subplot(337)
            plt.axis("off")
            rgb = obs[0, :3, :: self.du_scale, :: self.du_scale].permute(1, 2, 0)
            plt.imshow(rgb.cpu().numpy())
            plt.subplot(338)
            plt.imshow(depth[0].cpu().numpy())
            plt.axis("off")
            plt.subplot(339)
            seg = np.zeros_like(depth[0].cpu().numpy())
            for i in range(4, obs_channels):
                seg += (i - 4) * obs[0, i].cpu().numpy()
                print("class =", i, np.sum(obs[0, i].cpu().numpy()), "pts")
            plt.imshow(seg)
            plt.axis("off")
            plt.show()

            print("Non semantic channels =", MC.NON_SEM_CHANNELS)
            print("map shape =", current_map.shape)
            breakpoint()

        # if self.must_explore_close:
        #     current_map[:, MC.EXPLORED_MAP] = (
        #         current_map[:, MC.EXPLORED_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
        #     )
        #     current_map[:, MC.OBSTACLE_MAP] = (
        #         current_map[:, MC.OBSTACLE_MAP] * current_map[:, MC.BEEN_CLOSE_MAP]
        #     )

        return current_map, current_pose, extras

    def _update_global_map_and_pose_for_env(
        self,
        e: int,
        local_map: Tensor,
        global_map: Tensor,
        local_pose: Tensor,
        global_pose: Tensor,
        lmb: Tensor,
        origins: Tensor,
    ):
        """Update global map and pose and re-center local map and pose for a
        particular environment.
        """
        global_map[e, :, lmb[e, 0] : lmb[e, 1], lmb[e, 2] : lmb[e, 3]] = local_map[e]
        global_pose[e] = local_pose[e] + origins[e]
        mu.recenter_local_map_and_pose_for_env(
            e,
            local_map,
            global_map,
            local_pose,
            global_pose,
            lmb,
            origins,
            self.map_size_parameters,
        )

    def _get_map_features(self, local_map: Tensor, global_map: Tensor) -> Tensor:
        """Get global and local map features.

        Arguments:
            local_map: local map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
            global_map: global map of shape
             (batch_size, MC.NON_SEM_CHANNELS + num_sem_categories, M * ds, M * ds)

        Returns:
            map_features: semantic map features of shape
             (batch_size, 2 * MC.NON_SEM_CHANNELS + num_sem_categories, M, M)
        """
        map_features_channels = 2 * MC.NON_SEM_CHANNELS + self.num_sem_categories

        map_features = torch.zeros(
            local_map.size(0),
            map_features_channels,
            self.local_map_size,
            self.local_map_size,
            device=local_map.device,
            dtype=local_map.dtype,
        )

        # Local obstacles, explored area, and current and past position
        map_features[:, 0 : MC.NON_SEM_CHANNELS, :, :] = local_map[
            :, 0 : MC.NON_SEM_CHANNELS, :, :
        ]
        # Global obstacles, explored area, and current and past position
        map_features[
            :, MC.NON_SEM_CHANNELS : 2 * MC.NON_SEM_CHANNELS, :, :
        ] = nn.MaxPool2d(self.global_downscaling)(
            global_map[:, 0 : MC.NON_SEM_CHANNELS, :, :]
        )
        # Local semantic categories
        map_features[:, 2 * MC.NON_SEM_CHANNELS :, :, :] = local_map[
            :, MC.NON_SEM_CHANNELS :, :, :
        ]

        if debug_maps:
            plt.subplot(131)
            plt.imshow(local_map[0, 7])  # second object = cup
            plt.subplot(132)
            plt.imshow(local_map[0, 6])  # first object = chair
            # This is the channel in MAP FEATURES mode
            plt.subplot(133)
            plt.imshow(map_features[0, 12])
            plt.show()

        return map_features.detach()
