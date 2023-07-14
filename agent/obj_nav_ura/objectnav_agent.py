# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import DataParallel

import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)

from navigation_planner.discrete_planner import DiscretePlanner
import home_robot.utils.depth as du
from home_robot.utils import rotation as ru

from .objectnav_agent_module import ObjectNavAgentModule

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
from pytorch3d.ops import sample_farthest_points
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
from .points_utils import show_points, rasterize_pc
from .points_manager import PointCloudManager
# For visualizing exploration issues
debug_frontier_map = False


class ObjectNavAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0):
        self.max_steps = config.AGENT.max_steps
        self.num_environments = config.NUM_ENVIRONMENTS
        if config.AGENT.panorama_start:
            self.panorama_start_steps = int(360 / config.ENVIRONMENT.turn_angle)
        else:
            self.panorama_start_steps = 0

        self._module = ObjectNavAgentModule(config)

        if config.NO_GPU:
            self.device = torch.device("cpu")
            self.module = self._module
        else:
            self.device_id = device_id
            self.device = torch.device(f"cuda:{self.device_id}")
            self._module = self._module.to(self.device)
            # Use DataParallel only as a wrapper to move model inputs to GPU
            self.module = DataParallel(self._module, device_ids=[self.device_id])

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior,
        )
        agent_radius_cm = config.AGENT.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.ENVIRONMENT.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.VISUALIZE,
            print_images=config.PRINT_IMAGES,
            dump_location=config.DUMP_LOCATION,
            exp_name=config.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
        )
        self.one_hot_encoding = torch.eye(
            config.AGENT.SEMANTIC_MAP.num_sem_categories, device=self.device
        )

        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.verbose = config.AGENT.PLANNER.verbose

        self._use_probability_map = config.AGENT.SEMANTIC_MAP.use_probability_map
        self.probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior

        self.pointcloud = PointCloudManager(config)
        self.show_point = False
    # ------------------------------------------------------------------
    # Inference methods to interact with vectorized simulation
    # environments
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        object_goal_category: torch.Tensor = None,
        start_recep_goal_category: torch.Tensor = None,
        end_recep_goal_category: torch.Tensor = None,
        nav_to_recep: torch.Tensor = None,
        camera_pose: torch.Tensor = None,
        detection_results: Dict = None,
        raw_obs: Observations = None,
    ) -> Tuple[List[dict], List[dict]]:
        """Prepare low-level planner inputs from an observation - this is
        the main inference function of the agent that lets it interact with
        vectorized environments.

        This function assumes that the agent has been initialized.

        Args:
            obs: current frame containing (RGB, depth, segmentation) of shape
             (num_environments, 3 + 1 + num_sem_categories, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
             of shape (num_environments, 3)
            object_goal_category: semantic category of small object goals
            start_recep_goal_category: semantic category of start receptacle goals
            end_recep_goal_category: semantic category of end receptacle goals
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)

        Returns:
            planner_inputs: list of num_environments planner inputs dicts containing
                obstacle_map: (M, M) binary np.ndarray local obstacle map
                 prediction
                sensor_pose: (7,) np.ndarray denoting global pose (x, y, o)
                 and local map boundaries planning window (gx1, gx2, gy1, gy2)
                goal_map: (M, M) binary np.ndarray denoting goal location
            vis_inputs: list of num_environments visualization info dicts containing
                explored_map: (M, M) binary np.ndarray local explored map
                 prediction
                semantic_map: (M, M) np.ndarray containing local semantic map
                 predictions
        """
        dones = torch.tensor([False] * self.num_environments)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        if object_goal_category is not None:
            object_goal_category = object_goal_category.unsqueeze(1)
        if start_recep_goal_category is not None:
            start_recep_goal_category = start_recep_goal_category.unsqueeze(1)
        if end_recep_goal_category is not None:
            end_recep_goal_category = end_recep_goal_category.unsqueeze(1)
        (
            goal_map,
            found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            seq_pointcloud,
        ) = self.module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            seq_object_goal_category=object_goal_category,
            seq_start_recep_goal_category=start_recep_goal_category,
            seq_end_recep_goal_category=end_recep_goal_category,
            seq_nav_to_recep=nav_to_recep,
            detection_results=[detection_results],
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()

        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if found_goal[e]:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
            elif self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                self.timesteps_before_goal_update[e] = self.goal_update_steps

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1
            for e in range(self.num_environments)
        ]

        if debug_frontier_map:
            import matplotlib.pyplot as plt

            plt.subplot(131)
            plt.imshow(self.semantic_map.get_frontier_map(e))
            plt.subplot(132)
            plt.imshow(frontier_map[e][0])
            plt.subplot(133)
            plt.imshow(self.semantic_map.get_goal_map(e))
            plt.show()

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]

        ################# pointcloud #################
        xyz = seq_pointcloud[:, -1] / 100 # in agent base frame [N, P, 3]
        cur_pose = np.concatenate([raw_obs.gps, raw_obs.compass])

        



    #     xyz_global = du.transform_pose_t(
    #         xyz, cur_pose , xyz.device
    #     )

    #     if self.downsample_pc_k > 0:
    #         xyz_global, _ = sample_farthest_points(xyz_global, K=self.downsample_pc_k)
 

    #     if self.pointcloud is None:
    #         self.pointcloud = xyz_global
    #     else:
    #         self.pointcloud = torch.cat([self.pointcloud, xyz_global], dim=1) # [N, P, 3]
        
        self.pointcloud.add_points(points=xyz, agent_pose=cur_pose)

        if self.show_point:
            self.pointcloud.visualize()
    
            self.pointcloud.rasterize(camera_pose=raw_obs.camera_pose,
                                    agent_pose=cur_pose,
                                    lmb=self.semantic_map.lmb)
        ################# end pointcloud #################
        if self.visualize:
            vis_inputs = [
                {
                    "explored_map": self.semantic_map.get_explored_map(e),
                    "semantic_map": self.semantic_map.get_semantic_map(e),
                    "been_close_map": self.semantic_map.get_been_close_map(e),
                    "timestep": self.timesteps[e],
                }
                for e in range(self.num_environments)
            ]
        else:
            vis_inputs = [{} for e in range(self.num_environments)]

        return planner_inputs, vis_inputs

    def reset_vectorized(self):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.planner.reset()

    def reset_vectorized_for_env(self, e: int):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.planner.reset()

    # ---------------------------------------------------------------------
    # Inference methods to interact with the robot or a single un-vectorized
    # simulation environment
    # ---------------------------------------------------------------------

    def reset(self):
        """Initialize agent state."""
        self.reset_vectorized()
        self.planner.reset()

    def get_nav_to_recep(self):
        return None

    def act(self, obs: Observations) -> Tuple[DiscreteNavigationAction, Dict[str, Any]]:
        """Act end-to-end."""
        # t0 = time.time()

        # 1 - Obs preprocessing
        (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
            detection_results,
        ) = self._preprocess_obs(obs)

        # t1 = time.time()
        # print(f"[Agent] Obs preprocessing time: {t1 - t0:.2f}")

        # 2 - Semantic mapping + policy
        planner_inputs, vis_inputs = self.prepare_planner_inputs(
            obs_preprocessed,
            pose_delta,
            object_goal_category=object_goal_category,
            start_recep_goal_category=start_recep_goal_category,
            end_recep_goal_category=end_recep_goal_category,
            camera_pose=camera_pose,
            nav_to_recep=self.get_nav_to_recep(),
            detection_results=detection_results,
            raw_obs=obs,
        )

        # t2 = time.time()
        # print(f"[Agent] Semantic mapping and policy time: {t2 - t1:.2f}")

        # 3 - Planning
        closest_goal_map = None
        short_term_goal = None
        dilated_obstacle_map = None
        if planner_inputs[0]["found_goal"]:
            self.episode_panorama_start_steps = 0
        if self.timesteps[0] < self.episode_panorama_start_steps:
            action = DiscreteNavigationAction.TURN_RIGHT
        elif self.timesteps[0] > self.max_steps:
            action = DiscreteNavigationAction.STOP
        else:
            (
                action,
                closest_goal_map,
                short_term_goal,
                dilated_obstacle_map,
            ) = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                timestep=self.timesteps[0],
                debug=self.verbose,
            )

        # t3 = time.time()
        # print(f"[Agent] Planning time: {t3 - t2:.2f}")
        # print(f"[Agent] Total time: {t3 - t0:.2f}")
        # print()

        vis_inputs[0]["goal_name"] = obs.task_observations["goal_name"]
        if self.visualize:
            vis_inputs[0]["semantic_frame"] = obs.task_observations["semantic_frame"]
            vis_inputs[0]["closest_goal_map"] = closest_goal_map
            vis_inputs[0]["third_person_image"] = obs.third_person_image
            vis_inputs[0]["short_term_goal"] = None
            vis_inputs[0]["dilated_obstacle_map"] = dilated_obstacle_map
            vis_inputs[0]["probabilistic_map"] = self.semantic_map.get_probability_map(0)

        info = {**planner_inputs[0], **vis_inputs[0]}
        info["entropy"] = self.semantic_map.get_probability_map_entropy(0)
        return action, info

    def _preprocess_obs(self, obs: Observations):
        """Take a home-robot observation, preprocess it to put it into the correct format for the
        semantic map.
        Note: obs is a single observation, not a batch of observations.
        """
        rgb = torch.from_numpy(obs.rgb).to(self.device)
        depth = (
            torch.from_numpy(obs.depth).unsqueeze(-1).to(self.device) * 100.0
        )  # m to cm
            
        semantic = np.full_like(obs.semantic, 4)
        obj_goal_idx, start_recep_idx, end_recep_idx = 1, 2, 3
        semantic[obs.semantic == obs.task_observations["object_goal"]] = obj_goal_idx
        if "start_recep_goal" in obs.task_observations:
            semantic[
                obs.semantic == obs.task_observations["start_recep_goal"]
            ] = start_recep_idx
        if "end_recep_goal" in obs.task_observations:
            semantic[
                obs.semantic == obs.task_observations["end_recep_goal"]
            ] = end_recep_idx
        semantic = self.one_hot_encoding[torch.from_numpy(semantic).to(self.device)]
        obs_preprocessed = torch.cat([rgb, depth, semantic], dim=-1).unsqueeze(0)
        obs_preprocessed = obs_preprocessed.permute(0, 3, 1, 2)

        detection_results = None
        if self._use_probability_map:
            detection_results = {'scores':torch.tensor(obs.task_observations['instance_scores']).unsqueeze(0),
                                 'classes':torch.tensor(obs.task_observations['instance_classes']).unsqueeze(0),
                                 'masks':torch.tensor(obs.task_observations['masks']).unsqueeze(0)}
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = torch.tensor(
            pu.get_rel_pose_change(curr_pose, self.last_poses[0])
        ).unsqueeze(0)
        self.last_poses[0] = curr_pose
        object_goal_category = None
        end_recep_goal_category = None
        if (
            "object_goal" in obs.task_observations
            and obs.task_observations["object_goal"] is not None
        ):
            if self.verbose:
                print("object goal =", obs.task_observations["object_goal"])
            object_goal_category = torch.tensor(obj_goal_idx).unsqueeze(0)
        start_recep_goal_category = None
        if (
            "start_recep_goal" in obs.task_observations
            and obs.task_observations["start_recep_goal"] is not None
        ):
            if self.verbose:
                print("start_recep goal =", obs.task_observations["start_recep_goal"])
            start_recep_goal_category = torch.tensor(start_recep_idx).unsqueeze(0)
        if (
            "end_recep_goal" in obs.task_observations
            and obs.task_observations["end_recep_goal"] is not None
        ):
            if self.verbose:
                print("end_recep goal =", obs.task_observations["end_recep_goal"])
            end_recep_goal_category = torch.tensor(end_recep_idx).unsqueeze(0)
        goal_name = [obs.task_observations["goal_name"]]

        camera_pose = obs.camera_pose
        if camera_pose is not None:
            camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        return (
            obs_preprocessed,
            pose_delta,
            object_goal_category,
            start_recep_goal_category,
            end_recep_goal_category,
            goal_name,
            camera_pose,
            detection_results,
        )