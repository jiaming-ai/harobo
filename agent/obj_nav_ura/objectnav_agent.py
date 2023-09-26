# Adapted from https://github.com/facebookresearch/home-robot

from typing import Any, Dict, List, Tuple
from collections import deque
from datetime import datetime
import os
import numpy as np
import torch
from torch.nn import DataParallel
import home_robot.utils.pose as pu
from home_robot.core.abstract_agent import Agent
from home_robot.core.interfaces import DiscreteNavigationAction, Observations, ContinuousNavigationAction
from mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
import trimesh.transformations as tra
from .objectnav_agent_module import ObjectNavAgentModule
import numpy as np
import torch
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
from .ig_rendering.ig_renderer import IGRender
from pytorch3d.ops import sample_farthest_points
from navigation_planner.fmm_planner import FMMPlanner
from navigation_planner.discrete_planner import add_boundary, remove_boundary

# from .pn_agent import PNAgent
from igp_net.predictor import IGPredictor
from .ig_2d_ray_casting.ray_casting import IGRayCasting

# For visualizing exploration issues
debug_frontier_map = False
debug_info_gain = False
class ObjectNavAgent(Agent):
    """Simple object nav agent based on a 2D semantic map"""

    # Flag for debugging data flow and task configuraiton
    verbose = False

    def __init__(self, config, device_id: int = 0, **kwargs):
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

        self.visualize = config.VISUALIZE or config.PRINT_IMAGES or kwargs.get('visualize',False)
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
        self.use_FBE_policy = kwargs.get('use_FBE_policy',False) or kwargs.get('eval_rl_nav',False) # for evaluation
        if self.use_FBE_policy: 
            from navigation_planner.discrete_planner_fbe import DiscretePlanner
        else:
            from navigation_planner.discrete_planner import DiscretePlanner
            
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

        
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.episode_panorama_start_steps = None
        self.last_poses = None
        self.verbose = config.AGENT.PLANNER.verbose

        self.use_probability_map = config.AGENT.SEMANTIC_MAP.use_probability_map
        self.probability_prior=config.AGENT.SEMANTIC_MAP.probability_prior

        self.num_high_goals = 1
        self.max_ur_goal_dist_change = config.AGENT.IG_PLANNER.max_ur_goal_dist_change_cm / config.AGENT.SEMANTIC_MAP.map_resolution
        self.ur_dist_reach_threshold = config.AGENT.IG_PLANNER.ur_dist_reach_threshold_cm / config.AGENT.SEMANTIC_MAP.map_resolution
        self.agent_cell_radius = agent_cell_radius
        self.total_look_around_steps = config.AGENT.IG_PLANNER.total_look_around_steps
        self.turn_angle_rad = config.AGENT.IG_PLANNER.turn_angle_rad
        self.goal_update_steps = config.AGENT.IG_PLANNER.goal_update_steps
        self.max_num_changed_cells = config.AGENT.IG_PLANNER.max_num_changed_cells
        self.max_num_changed_promising_cells = config.AGENT.IG_PLANNER.max_num_changed_promising_cells
        self.info_gain_alpha = config.AGENT.IG_PLANNER.info_gain_alpha
        self.ur_obstacle_dialate_radius = config.AGENT.IG_PLANNER.ur_obstacle_dialate_radius
        self.ur_dist_obstacle_dialate_radius = config.AGENT.IG_PLANNER.ur_dist_obstacle_dialate_radius
        self.util_lambda = config.AGENT.IG_PLANNER.util_lambda
        self.random_ur_goal = config.AGENT.IG_PLANNER.random_ur_goal

        """
        UR Exploration:
            0: go to ur goal (look around point)
            1: look around
        
        Point Goal Navigation:
            2: object identified, go to object
        
        """
        self._state = np.ones(self.num_environments) # start with look around
        self._ur_goal_dist = np.zeros(self.num_environments)
        self._global_hgoal_pose = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device) # (num_envs, num_goals, 2)
        self._ur_local_goal_coords = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device) # (num_envs, num_goals, 2)
        self._force_goal_update_once = np.full(self.num_environments, False)
        self._look_around_steps = np.zeros(self.num_environments)
        self._num_explored_grids = np.zeros(self.num_environments)
        self._num_promising_grids = np.zeros(self.num_environments)

        self.save_info_gain_data = kwargs.get('collect_data',False) # for training

        self.ig_renderer = None

        self.ig_predictor_type = config.AGENT.IG_PLANNER.ig_predictor_type
        self.utility_exp = config.AGENT.IG_PLANNER.utility_exp
        if self.ig_predictor_type == 'oig':
            igp_model_dir = config.AGENT.IG_PLANNER.igp_model_dir
            self.ig_predictor = IGPredictor(igp_model_dir,self.device)
        else:
            self.max_render_loc = 150
            self.uniform_sampling = True
            if self.ig_predictor_type == 'ray_casting':
                self.ig_ray_casting = IGRayCasting(config,device=self.device)

        #### stuck detection ####
        self.pre_action_deque_len = 6
        self.replan_ur_goal_if_stuck = config.AGENT.IG_PLANNER.replan_ur_goal_if_stuck
        self._pre_actions_taken = [deque(maxlen=self.pre_action_deque_len) for _ in range(self.num_environments)]
        self._pre_pose = [deque(maxlen=self.pre_action_deque_len) for _ in range(self.num_environments)]

    def force_update_high_goal(self,e):
        self._force_goal_update_once[e] = True

    def init_ig_renderer(self,obs):
        """
        we need obs to init the dr planner
        """
        if 'camera_pose' in obs.__dict__:
            angles = tra.euler_from_matrix(obs.camera_pose[:3, :3], "rzyx")
            self.camera_tilt = - angles[1]

            # Get the camera height
            self.camera_height = obs.camera_pose[2, 3]
        else:
            # otherwise use the default values
            self.camera_tilt = 0
            self.camera_height = self.config.ENVIRONMENT.camera_height

        # currently only support a single environment
        self.ig_renderer = IGRender(self.camera_tilt, self.camera_height, self.config, self.device)

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
       
        # update_global = torch.tensor(
        #     [
        #         (self.timesteps_before_goal_update[e] == 0) or (self.force_goal_update_once)
        #         for e in range(self.num_environments)
        #     ]
        # )

        # update the map every single step
        update_global = torch.tensor(
            [
                True for e in range(self.num_environments)
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
            seq_extras,
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
        
        extras = seq_extras[-1]
        goal_map = goal_map.squeeze(1).cpu().numpy()
        found_goal = found_goal.squeeze(1).cpu()

        ig_vis_list = [None for e in range(self.num_environments)]
        ig_time_list = [None for e in range(self.num_environments)]
        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            # find object goal
            if found_goal[e] or self.use_FBE_policy:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                # calculate the global goal pose
                xs, ys = goal_map[e].nonzero()
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                pos = torch.tensor([xc, yc], device=self.device).unsqueeze(0)
                self._global_hgoal_pose[e] = self.semantic_map.local_map_coords_to_hab_world_frame(e, pos)

                self._state[e] = 2 # go to object goal
            
            # if not find object goal and use a IG planner
            else:
                # if need to update high goal
                if self._check_n_update_need_replan_ur_goal(e):
                    import time
                    start = time.time()
                    if self.ig_predictor_type in ['oig','argmax']:
                        if self.ig_predictor_type == 'oig':
                            pred_ig, ig_vis = self._compute_info_gains_igp(e)
                        else:
                            pred_ig, ig_vis = self._compute_info_gains_by_argmax(e)
                        utility_map,local_goal_coords,dist = self._select_goal_igp(e, pred_ig)
                        self._ur_goal_dist[e] = dist
                        self._ur_local_goal_coords[e] = local_goal_coords
                        self._global_hgoal_pose[e] = self.semantic_map.local_map_coords_to_hab_world_frame(e, local_goal_coords)
                        goal_map_e = self._get_goal_map(local_goal_coords)
                        if self.visualize:
                            ig_vis['utility'] = render_plt_image(utility_map)
                            ig_vis_list[e] = ig_vis
                    else:
                        if self.ig_predictor_type == 'ray_casting':
                            local_goal_coords, global_pose, info = self._compute_info_gains_ray_casting(e)
                        elif self.ig_predictor_type == 'rendering':
                            local_goal_coords, global_pose, info = self._compute_info_gains_by_render(e)
                        else:
                            raise NotImplementedError
                        if local_goal_coords is None:
                            continue
                        dist_map = self._get_dist_map(e)
                        selected_goal_coords, selected_info, selected_idx = self._select_goal(dist_map,info,local_goal_coords)
                        self._global_hgoal_pose[e] = global_pose[selected_idx] # selected global pos
                        self._ur_local_goal_coords[e] = local_goal_coords[selected_idx] # selected local pos

                        # local_goal_coords = self._ur_local_goal_coords[e]
                        self._ur_goal_dist[e] = dist_map[local_goal_coords[0,0],local_goal_coords[0,1]].item() # we only consider the first goal
                        goal_map_e = self._get_goal_map(selected_goal_coords)
                    oig_time = time.time() - start
                    ig_time_list[e] = oig_time

                    self._state[e] = 0 # go to the ur goal
                    self.semantic_map.update_global_goal_for_env(e, goal_map_e)
                    
                # not find obj and no need to update high goal
                # we just need to transform the high goal to account for the robot's movement
                else:
                    goal_coords = self.semantic_map.hab_world_to_map_local_frame(e, self._global_hgoal_pose[e])
                    self._ur_local_goal_coords[e] = goal_coords
                    goal_map = self._get_goal_map(goal_coords)
                    self.semantic_map.update_global_goal_for_env(e, goal_map)
                

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        
        
        ######################### state transition #########################
        ### this must be after goal update
        reach_goal = self._ur_goal_dist <= self.ur_dist_reach_threshold
        for e in range(self.num_environments):
            # goto object goal if found goal
            # otherwise, if reach ur goal, look around
            # we also force look around at the beginning
            if self._state[e] == 0 \
                and (not found_goal[e]) \
                and (self.timesteps[e] < self.total_look_around_steps or reach_goal[e]):
                self._state[e] = 1  # look around

            
        ####################################################################

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
                "global_goal_pose": self._global_hgoal_pose[e].cpu().numpy(),
            }
            for e in range(self.num_environments)
        ]

        vis_inputs = [
            {
                "timestep": self.timesteps[e],
                "checking_area": extras["checking_area"][e] if "checking_area" in extras else None,
                "ig_time": ig_time_list[e],
            }
            for e in range(self.num_environments)
        ]

        if self.visualize:
            for e in range(self.num_environments):
                vis_inputs[e].update(
                    {
                        "explored_map": self.semantic_map.get_explored_map(e),
                        "semantic_map": self.semantic_map.get_semantic_map(e),
                        "been_close_map": self.semantic_map.get_been_close_map(e),
                        "ig_vis": ig_vis_list[e],
                    })

        return planner_inputs, vis_inputs

    def reset_vectorized(self,episodes=None):
        """Initialize agent state."""
        self.timesteps = [0] * self.num_environments
        self.timesteps_before_goal_update = [0] * self.num_environments
        self.last_poses = [np.zeros(3)] * self.num_environments
        self.semantic_map.init_map_and_pose()
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.planner.reset()

        self._state = np.ones(self.num_environments)
        self._ur_goal_dist = np.zeros(self.num_environments)
        self._force_goal_update_once = np.full(self.num_environments, False)
        self._global_hgoal_pose = torch.zeros((self.num_environments,self.num_high_goals, 2), device=self.device)
        self._ur_local_goal_coords = torch.zeros((self.num_environments,self.num_high_goals, 2), 
                                                 dtype=torch.long,
                                                 device=self.device)
        self._look_around_steps = np.zeros(self.num_environments)
        self._num_explored_grids = np.zeros(self.num_environments)
        self._num_promising_grids = np.zeros(self.num_environments)
        self._pre_actions_taken = [deque(maxlen=self.pre_action_deque_len) for _ in range(self.num_environments)]
        self._pre_pose = [deque(maxlen=self.pre_action_deque_len) for _ in range(self.num_environments)]

        if self.save_info_gain_data:
            if episodes is None: 
                now = datetime.now()
                self.save_info_gain_data_dir = "data/info_gain/" + now.strftime("%Y_%m_%d_%H_%M_%S")
                
            else:
                scene_id = episodes[0].scene_id.split("/")[-1].split(".")[0]
                eps_id = episodes[0].episode_id
                self.save_info_gain_data_dir = f'data/info_gain/{scene_id}/{eps_id}'
                
            os.makedirs(self.save_info_gain_data_dir, exist_ok=True)
                

    def reset_vectorized_for_env(self, e: int, episode=None):
        """Initialize agent state for a specific environment."""
        self.timesteps[e] = 0
        self.timesteps_before_goal_update[e] = 0
        self.last_poses[e] = np.zeros(3)
        self.semantic_map.init_map_and_pose_for_env(e)
        self.episode_panorama_start_steps = self.panorama_start_steps
        self.planner.reset()
        self._state[e] = 1
        self._ur_goal_dist[e] = 0
        self._force_goal_update_once[e] = False
        self._global_hgoal_pose[e] = torch.zeros((self.num_high_goals, 2), device=self.device)
        self._ur_local_goal_coords[e] = torch.zeros((self.num_high_goals, 2), device=self.device).long()
        self._look_around_steps[e] = 0
        self._num_explored_grids[e] = 0
        self._num_promising_grids[e] = 0
        self._pre_actions_taken[e] = deque(maxlen=self.total_look_around_steps * 2)
        self._pre_pose[e] = deque(maxlen=self.total_look_around_steps * 2)

        if self.save_info_gain_data:
            if episode is None:
                now = datetime.now()
                self.save_info_gain_data_dir = "data/info_gain/" + now.strftime("%Y_%m_%d_%H_%M_%S")
                
            else:
                scene_id = episode.scene_id.split("/")[-1].split(".")[0]
                eps_id = episode.episode_id
                self.save_info_gain_data_dir = f'data/info_gain/{scene_id}/{eps_id}'
                
            os.makedirs(self.save_info_gain_data_dir, exist_ok=True)

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
        if self.ig_renderer is None and self.ig_predictor_type == 'rendering':
            self.init_ig_renderer(obs)

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
        # if planner_inputs[0]["found_goal"]:
        #     self.episode_panorama_start_steps = 0

        if self.timesteps[0] > self.max_steps:
            action = DiscreteNavigationAction.STOP
        
        # state is 1: look around
        elif self._state[0] == 1:
            
            # TODO: we should use more sophisticated policy here. Currently using a simple one
            action = ContinuousNavigationAction(np.array([0.,0.,self.turn_angle_rad]))
            # action = DiscreteNavigationAction.TURN_RIGHT
            # self.timesteps_before_goal_update[0] += 1 # make sure we don't update the goal during look around
            self._look_around_steps[0] += 1
            # we must change state directly here, otherwise the urp planner will not be called
            if self._look_around_steps[0] >= self.total_look_around_steps:
                self._state[0] = 0 # go to ur goal
                self._look_around_steps[0] = 0 # reset look around steps
                self.timesteps_before_goal_update[0] = 0 # relan for the next ur goal
        
        # state is 0 or 2: go to point goal
        
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
            

        # if we are moving towards ur goal, we should not call STOP
        if self._state[0] == 0 and action == DiscreteNavigationAction.STOP:
            # change to look around, and start turning
            self._state[0] = 1 # look around
            self._look_around_steps[0] += 1
            action = ContinuousNavigationAction(np.array([0.,0.,self.turn_angle_rad]))

        act_idx = action.value if isinstance(action, DiscreteNavigationAction) else -1
        self._pre_actions_taken[0].append(act_idx)
        self._pre_pose[0].append(obs.gps)


        # # test rl planner
        # depth = torch.from_numpy(obs.depth).unsqueeze(0)
        # goal = torch.tensor([-1.,0.]) # debug
        # agent_pose = torch.from_numpy(np.concatenate([obs.gps,obs.compass],axis=0))
        # action = self.pn_agent.plan(depth,goal,agent_pose)

        # t3 = time.time()
        # print(f"[Agent] Planning time: {t3 - t2:.2f}")
        # print(f"[Agent] Total time: {t3 - t0:.2f}")
        # print()

        vis_inputs[0]["goal_name"] = obs.task_observations["goal_name"]
        vis_inputs[0]["exp_coverage"] = self.semantic_map.get_exp_coverage_area(0)
        vis_inputs[0]["close_coverage"] = self.semantic_map.get_close_coverage_area(0)

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

    #####################################################################
    # Helper methods for UR exploration
    #####################################################################
    def _select_goal_igp(self,e:int, ig_map):
        """
        Calculate the (local) utility map
        args: 
            ig_map: global info gain map 
        """
        dist_map = self._get_dist_map(e)
        lmb = self.semantic_map.lmb[e]
        local_ig_map = ig_map[lmb[0]:lmb[1],lmb[2]:lmb[3]]
        
        # we want to prioritize the grids that are close to the current location
        # and have high info gain
        # U = I*e^(-lambda * d)
        # smaller lambda means we want to prioritize the grids that are close to the current location
        if self.utility_exp:
            utility_map = local_ig_map * torch.exp(-self.util_lambda * dist_map)
        else:
            dist_map[dist_map==0] = 10
            utility_map = local_ig_map / dist_map ** 0.7 * 10
            
        # select goal with max utility
        if not self.random_ur_goal:
            local_goal_idx = torch.argmax(utility_map)

        # select goal randomly from topk utility
        else:
            u, idx = torch.topk(utility_map.view(-1), 100)
            local_goal_idx = idx[torch.randint(0,20,(1,))]
        
        
        x = local_goal_idx // utility_map.shape[1]
        y = local_goal_idx % utility_map.shape[1]
        local_goal = torch.tensor([[x,y]]).to(self.device)
        
        return utility_map,local_goal,dist_map[x,y].item()
    
    def _get_dist_map(self,e:int) -> torch.tensor:
        """
        Calculate the (local) distance field (in cells) from the current location
        args:
            e: environment index
        returns:
            dist_map: distance field in cells

        """
        # NOTE: if we dialate too much, the obstacle may swallow the agent, and the dist map will be wrong
        dilated_obstacles = self.semantic_map.get_dialated_obstacle_map_local(e,self.ur_dist_obstacle_dialate_radius)
        agent_rad = self.agent_cell_radius

        # we need to make sure the agent is not inside the obstacle
        while True:
            traversible = 1 - dilated_obstacles
            start = self.semantic_map.get_local_coords(e)

            agent_rad += 1
            traversible[
                start[0] - agent_rad : start[0] + agent_rad + 1,
                start[1] - agent_rad : start[1] + agent_rad + 1,
            ] = 1
            traversible = add_boundary(traversible)
            vis_planner = FMMPlanner(traversible)
            curr_loc_map = np.zeros_like(traversible)

            # Update our location for finding the closest goal
            start = self.semantic_map.get_local_coords(e)
            curr_loc_map[start[0], start[1]] = 1
            # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
            vis_planner.set_multi_goal(curr_loc_map)
            fmm_dist = vis_planner.fmm_dist
            fmm_dist = remove_boundary(fmm_dist)
            fmm_dist[dilated_obstacles==1] = 10000

            # check if the agent is inside the obstacle
            if np.unique(fmm_dist).shape[0] > 10:
                return torch.from_numpy(fmm_dist).float().to(self.device)
    
    def _select_goal(self,dist_map:torch.tensor, 
                     infos: torch.tensor, 
                     locs: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Select the goal with the highest info gain and close to the current location
        args:
            dist_map: distance map, tensor of shape [H,W]
            infos: info gain at each locs, tensor of shape [N]
            locs: locations of the goals, tensor of shape [N,2] (in local map coords)
        """
        
        utility = infos / dist_map[locs[:,0],locs[:,1]]**0.7 * 100
        utility_sorted, idx = torch.sort(utility, descending=True)
        locs = locs[idx]
        return locs[:self.num_high_goals], utility_sorted[:self.num_high_goals], idx[:self.num_high_goals]
    
    def _get_goal_map(self,locs:torch.tensor) -> np.ndarray:
        """
        Convert the goal locations to a goal map that can be used by the planner
        args:
            locs: locations of the goals, tensor of shape [N,2] (in local map coords)
        returns:
            goal_map: goal map, numpy array of shape [H,W]
        """

        goal_map = torch.zeros([self.semantic_map.local_map_size,self.semantic_map.local_map_size])
        goal_map[locs[:,0],locs[:,1]] = 1        

        return goal_map.cpu().numpy()

    def _check_n_update_need_replan_ur_goal(self, e:int) -> bool:
        """
        decide whether to replan. We also update ur_goal_dist here.
        Replan if:
            1. if the selected goal is occupaid by obstacles
            2. if the map changes too much (so there could be more interesting locs to explore?)
            3. if the distance to selected goal changed too much, because of newly 
            observed obstacles. (so better to explore nearby first?)
            4. if force_update or timesteps_before_goal_update is reached, or we have traveled too far
            5. if the agent is stuck
        
        """
        # no need to replan if we are looking around
        if self._state[e] == 1:
            return False
        
        replan = False

        # case 5
        if self._check_is_stuck(e):
            replan = True

        # case 4
        if self.timesteps_before_goal_update[e] == 0 or self._force_goal_update_once[e]:
            replan = True
        
        # case 2
        num_exp_grid = self.semantic_map.get_num_explored_cells(e)
        num_changed_cells = num_exp_grid - self._num_explored_grids[e]
        num_promising_cells = self.semantic_map.get_num_promising_cells(e)
        num_changed_promising_cells = abs(num_promising_cells - self._num_promising_grids[e])
        if num_changed_cells > self.max_num_changed_cells:
            replan = True
        if num_changed_promising_cells > self.max_num_changed_promising_cells:
            replan = True
        
        # case 1, 3
        goal_coords = self.semantic_map.hab_world_to_map_local_frame(e, self._global_hgoal_pose[e])
        dist_map = self._get_dist_map(e)
        goal_dist = dist_map[goal_coords[0,0],goal_coords[0,1]].item() # we only consider the first goal
        
        # if new goal dist changed too much, replan
        if ( goal_dist - self._ur_goal_dist[e] ) > self.max_ur_goal_dist_change: 
            replan = True

        # update internal states
        if replan:
            self.timesteps_before_goal_update[e] = self.goal_update_steps
            self._force_goal_update_once[e] = False
            self._num_explored_grids[e] = num_exp_grid
            self._num_promising_grids[e] = num_promising_cells
            
        else:
            self._ur_goal_dist[e] = goal_dist
            # timestep to replan for the ur goal
            self.timesteps_before_goal_update = [
                self.timesteps_before_goal_update[e] - 1
                for e in range(self.num_environments)
            ]

        return replan

    def _get_promising_dist(self, e:int) -> np.ndarray:
        """
        get the distance map of 'promising' areas for the current environment
        """
        dilated_obstacles = self.semantic_map.get_dialated_obstacle_map_local(e,1) 
        traversible = 1 - dilated_obstacles

        #################
        # first we find areas that are close to the promising areas and traversible
        planner = FMMPlanner(np.ones_like(traversible))
        promising_map = self.semantic_map.get_promising_map(e) 
        # Plan to the goal mask
        planner.set_multi_goal(promising_map)

        # Now mask out anything here based on distance to the goal mask
        dist_map = planner.fmm_dist * traversible
        dist_map[dist_map == 0] = dist_map.max()

        navigable_goal_map = dist_map < 8
        ##################

        ##################
        # then we compute the distance map to the navigable goal map

        traversible = add_boundary(traversible)
        vis_planner = FMMPlanner(traversible)
        navigable_goal_map = add_boundary(navigable_goal_map)
     
        vis_planner.set_multi_goal(navigable_goal_map)
        fmm_dist = vis_planner.fmm_dist
        fmm_dist = remove_boundary(fmm_dist)
        fmm_dist[dilated_obstacles==1] = 10000
        
        return torch.from_numpy(fmm_dist).float().to(self.device)
        
    def _compute_info_gains_igp(self,e):
        # exp_pos_all = self.semantic_map.get_explored_locs(e) # [N, 2]
        # if exp_pos_all.shape[0] == 0:
        #     return None, None, None
        # n_p = exp_pos_all.shape[0]
        # print(f'Num of locations: {n_p}')
        # if n_p > self._max_render_loc:
        #     if self._uniform_sampling:
        #         rng = np.random.default_rng()
        #         arr = np.arange(n_p)
        #         rng.shuffle(arr)
        #         exp_pos = exp_pos_all[arr[:self._max_render_loc]]
        #     else:
        #         exp_pos = sample_farthest_points(exp_pos_all.unsqueeze(0), K=self._max_render_loc)[0] # [1, N, 2]
        #         exp_pos = exp_pos.squeeze(0) # [N, 2]

        # global_exp_pos_map_frame = self.semantic_map.hab_world_to_map_global_frame(e, exp_pos)

        voxel = self.semantic_map.get_global_voxel(e)
        if not self.use_probability_map:
            voxel[~voxel.isinf()] = -13 # mark all occupaid as 0 prob
        pred = self.ig_predictor.predict(voxel)
        # obstacle = self.semantic_map.global_map[e,0] > 0 # [M x M]
        obstacle = self.semantic_map.get_dialated_obstacle_map_global(e,self.ur_obstacle_dialate_radius) # [M x M]
        
        # we add obstacles from collision map
        collision_map = torch.from_numpy(self.planner.collision_map).to(self.device)
        obstacle = torch.logical_or(obstacle, collision_map)
        pred[obstacle.repeat(2,1,1)] = 0
        
        pred_ig = pred[0] + self.info_gain_alpha * pred[1] / self.ig_predictor.i_s_weight
        # info_at_locs = pred_ig[global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] # [N_locs]

        # info_sorted, idx = torch.sort(info_at_locs, descending=True)
        # local_exp_pos_map_frame = self.semantic_map.hab_world_to_map_local_frame(e, exp_pos)
        # local_exp_pos_map_frame = local_exp_pos_map_frame[idx] # sorted
        # local_exp_pos_map_frame_sorted = local_exp_pos_map_frame.clone()
        # exp_pos_sorted = exp_pos[idx]

        vis = {}
        if self.visualize:
            vis['cs'] = render_plt_image(pred[0])
            vis['is'] = render_plt_image(pred[1]/5) 
            vis['ig'] = render_plt_image(pred_ig)

        # return local_exp_pos_map_frame_sorted, exp_pos_sorted, info_sorted, pred_ig, vis
        return pred_ig, vis
    
    def _compute_info_gains_by_render(self,e):
        """
        compute the info gains for 
        
        """
        exp_pos_all = self.semantic_map.get_explored_locs(e) # [N, 2]
        if exp_pos_all.shape[0] == 0:
            return None, None, None
        n_p = exp_pos_all.shape[0]
        print(f'Num of locations: {n_p}')
        if n_p > self.max_render_loc:
            if self.uniform_sampling:
                rng = np.random.default_rng()
                arr = np.arange(n_p)
                rng.shuffle(arr)
                exp_pos = exp_pos_all[arr[:self.max_render_loc]]
            else:
                exp_pos = sample_farthest_points(exp_pos_all.unsqueeze(0), K=self.max_render_loc)[0] # [1, N, 2]
                exp_pos = exp_pos.squeeze(0) # [N, 2]
        
            ############################
            # sample more points close to promising areas
            if self.save_info_gain_data:
                promising_map = self.semantic_map.get_promising_map(e) # [H, W]
                if promising_map.sum() > 0:
                    promising_dist_map = self._get_promising_dist(e) # [H, W]
                    local_exp_pos_map_frame = self.semantic_map.hab_world_to_map_local_frame(e, exp_pos_all)

                    promising_dist = promising_dist_map[local_exp_pos_map_frame[:,0], local_exp_pos_map_frame[:,1]] # [N]
                    close_to_promising_idx = promising_dist < 40 # 1m
                    total_close = close_to_promising_idx.sum().item()
                    close_len = min(50, total_close)
                    if close_len > 0:
                        rng = np.random.default_rng()
                        arr = np.arange(total_close)
                        rng.shuffle(arr)
                        exp_pos = torch.cat([exp_pos[:-close_len], exp_pos_all[close_to_promising_idx][arr[:close_len]]])
                
        else:
            exp_pos = exp_pos_all

        points, feats = self.semantic_map.get_global_pointcloud(e) # [P, 3], [P, 1]
        points = points.unsqueeze(0) # [1, P, 3]
        feats = feats.unsqueeze(0) # [1, P, 1]
        result = self.ig_renderer.cal_info_gains(points,feats,exp_pos)
        c_s, i_s = result['coverage_info'], result['promising_info'] # [N_locs], [N_locs]
        info_at_locs = (c_s + self.info_gain_alpha * i_s) # [N_locs]
        
        info_sorted, idx = torch.sort(info_at_locs, descending=True)
        local_exp_pos_map_frame = self.semantic_map.hab_world_to_map_local_frame(e, exp_pos)
        local_exp_pos_map_frame = local_exp_pos_map_frame[idx] # sorted
        local_exp_pos_map_frame_sorted = local_exp_pos_map_frame.clone()
        exp_pos_sorted = exp_pos[idx]

        if self.save_info_gain_data:
            if '_last_save_num_explored_grids' not in self.__dict__:
                # assume single env
                self._last_save_num_explored_grids = 0

            num_exp_grid = self.semantic_map.get_num_explored_cells(e)
            num_changed_cells = abs(num_exp_grid - self._last_save_num_explored_grids)
            # we only save the data if the number of changed cells is larger than 100
            # this is to avoid saving same data
            if num_changed_cells > 100:
                self._last_save_num_explored_grids = num_exp_grid
                global_exp_pos_map_frame = self.semantic_map.hab_world_to_map_global_frame(e, exp_pos).cpu()
                point_idx = self.semantic_map.get_global_pointcloud_flat_idx(e).squeeze(-1).cpu().to(torch.int32)
                p = feats[0].squeeze(-1).cpu().to(torch.float16) # save space
                c_s, i_s = result['raw_c_s'].cpu(), result['raw_i_s'].cpu() # [N_locs * len(theta_list))]
                theta_tensor = torch.tensor(result['theta_list'])
                torch.save([point_idx, p, global_exp_pos_map_frame, c_s,i_s,theta_tensor],
                        self.save_info_gain_data_dir+f'/{self.timesteps[0]}.pt')

        if debug_info_gain:
        ##################### visualize using global map
            global_exp_pos_map_frame = self.semantic_map.hab_world_to_map_global_frame(e, exp_pos)
            global_exp_pos_map_frame = global_exp_pos_map_frame[idx] # sorted

            global_map_color = torch.zeros_like(self.semantic_map.global_map[e,0])  # [H, W]
            global_map_color = global_map_color.unsqueeze(-1).repeat(1,1,3) # [H, W, 3]
            
            # mark first 5 points as red
            n_show = 20
            global_map_color[global_exp_pos_map_frame[:n_show,0], global_exp_pos_map_frame[:n_show,1], :] = \
                                            torch.tensor([1.,0.,0.]).to(global_map_color.device)
            # mark the last 5 points as green
            global_map_color[global_exp_pos_map_frame[-n_show:,0], global_exp_pos_map_frame[-n_show:,1], :] = \
                                            torch.tensor([0.,1.,0.]).to(global_map_color.device)
            # mark the rest as grey
            global_map_color[global_exp_pos_map_frame[n_show:-n_show,0], global_exp_pos_map_frame[n_show:-n_show,1], :] = \
                                            torch.tensor([0.5,0.5,0.5]).to(global_map_color.device)
            
            global_map_color = global_map_color.cpu().numpy()


            #################### visualize all sampled points
            global_map_show_pos = torch.zeros_like(self.semantic_map.global_map[e,0])
            global_map_show_pos[global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = 1.0


            #################### visualize local goal map top 5
            local_goal_map = torch.zeros_like(self.semantic_map.local_map[e,0])    # [H, W]
            local_exp_pos_map_frame = local_exp_pos_map_frame[:n_show] # first 5 points

            local_goal_map[local_exp_pos_map_frame[:,0], local_exp_pos_map_frame[:,1]] = 1.0

        #################### debug IGP
        # voxel = self.semantic_map.get_global_voxel(e)
        # pred = self.ig_predictor.predict(voxel)

        return local_exp_pos_map_frame_sorted, exp_pos_sorted, info_sorted
    
    def _compute_info_gains_by_argmax(self,e):
        """
        simply select the point with the highest probability
        """
        prob_map = self.semantic_map.get_global_probability_map_tensor(e) # [H, W]
        # exp_map = self.semantic_map.get_global_exp_map_tensor(e) # [H, W]
        # prob_map[exp_map==0] = 0 # we don't want to go to unexplored areas
        
        obstacle = self.semantic_map.get_dialated_obstacle_map_global(e,self.ur_obstacle_dialate_radius) # [M x M]
        
        # we add obstacles from collision map
        collision_map = torch.from_numpy(self.planner.collision_map).to(self.device)
        obstacle = torch.logical_or(obstacle, collision_map)
        prob_map[obstacle] = 0

        max_prob, idx = torch.max(prob_map.view(-1), dim=0)
        x = idx // prob_map.shape[1]
        y = idx % prob_map.shape[1]
        pred_ig = torch.zeros_like(prob_map)
        pred_ig[x,y] = 1.0

        return pred_ig, {}
        
    def _compute_info_gains_ray_casting(self,e):
        """
        compute the info gains for 
        
        """
        exp_pos_all = self.semantic_map.get_explored_locs(e) # [N, 2]
        if exp_pos_all.shape[0] == 0:
            return None, None, None
        n_p = exp_pos_all.shape[0]
        print(f'Num of locations: {n_p}')
        if n_p > self.max_render_loc:
            if self.uniform_sampling:
                rng = np.random.default_rng()
                arr = np.arange(n_p)
                rng.shuffle(arr)
                exp_pos = exp_pos_all[arr[:self.max_render_loc]]
            else:
                exp_pos = sample_farthest_points(exp_pos_all.unsqueeze(0), K=self.max_render_loc)[0] # [1, N, 2]
                exp_pos = exp_pos.squeeze(0) # [N, 2]
        
            ############################
            # sample more points close to promising areas
            if self.save_info_gain_data:
                promising_map = self.semantic_map.get_promising_map(e) # [H, W]
                if promising_map.sum() > 0:
                    promising_dist_map = self._get_promising_dist(e) # [H, W]
                    local_exp_pos_map_frame = self.semantic_map.hab_world_to_map_local_frame(e, exp_pos_all)

                    promising_dist = promising_dist_map[local_exp_pos_map_frame[:,0], local_exp_pos_map_frame[:,1]] # [N]
                    close_to_promising_idx = promising_dist < 40 # 1m
                    total_close = close_to_promising_idx.sum().item()
                    close_len = min(50, total_close)
                    if close_len > 0:
                        rng = np.random.default_rng()
                        arr = np.arange(total_close)
                        rng.shuffle(arr)
                        exp_pos = torch.cat([exp_pos[:-close_len], exp_pos_all[close_to_promising_idx][arr[:close_len]]])
                
        else:
            exp_pos = exp_pos_all

        points, feats = self.semantic_map.get_global_pointcloud(e) # [P, 3], [P, 1]
        points = points.unsqueeze(0) # [1, P, 3]
        feats = feats.unsqueeze(0) # [1, P, 1]
        c_s, i_s = self.ig_ray_casting.cal_info_gains(self.semantic_map.global_map[e,0].cpu().numpy(),
                                    self.semantic_map.global_map[e,1].cpu().numpy(),
                                    self.semantic_map.global_map[e,2].cpu().numpy(),
                                    self.semantic_map.hab_world_to_map_global_frame(e,exp_pos).cpu().numpy()
                                    )
        
        info_at_locs = (c_s + self.info_gain_alpha * i_s) # [N_locs]
        
        info_sorted, idx = torch.sort(info_at_locs, descending=True)
        local_exp_pos_map_frame = self.semantic_map.hab_world_to_map_local_frame(e, exp_pos)
        local_exp_pos_map_frame = local_exp_pos_map_frame[idx] # sorted
        local_exp_pos_map_frame_sorted = local_exp_pos_map_frame.clone()
        exp_pos_sorted = exp_pos[idx]

      

        if debug_info_gain:
        ##################### visualize using global map
            global_exp_pos_map_frame = self.semantic_map.hab_world_to_map_global_frame(e, exp_pos)
            global_exp_pos_map_frame = global_exp_pos_map_frame[idx] # sorted

            global_map_color = torch.zeros_like(self.semantic_map.global_map[e,0])  # [H, W]
            global_map_color = global_map_color.unsqueeze(-1).repeat(1,1,3) # [H, W, 3]
            
            # mark first 5 points as red
            n_show = 20
            global_map_color[global_exp_pos_map_frame[:n_show,0], global_exp_pos_map_frame[:n_show,1], :] = \
                                            torch.tensor([1.,0.,0.]).to(global_map_color.device)
            # mark the last 5 points as green
            global_map_color[global_exp_pos_map_frame[-n_show:,0], global_exp_pos_map_frame[-n_show:,1], :] = \
                                            torch.tensor([0.,1.,0.]).to(global_map_color.device)
            # mark the rest as grey
            global_map_color[global_exp_pos_map_frame[n_show:-n_show,0], global_exp_pos_map_frame[n_show:-n_show,1], :] = \
                                            torch.tensor([0.5,0.5,0.5]).to(global_map_color.device)
            
            global_map_color = global_map_color.cpu().numpy()


            #################### visualize all sampled points
            global_map_show_pos = torch.zeros_like(self.semantic_map.global_map[e,0])
            global_map_show_pos[global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = 1.0


            #################### visualize local goal map top 5
            local_goal_map = torch.zeros_like(self.semantic_map.local_map[e,0])    # [H, W]
            local_exp_pos_map_frame = local_exp_pos_map_frame[:n_show] # first 5 points

            local_goal_map[local_exp_pos_map_frame[:,0], local_exp_pos_map_frame[:,1]] = 1.0

        #################### debug IGP
        # voxel = self.semantic_map.get_global_voxel(e)
        # pred = self.ig_predictor.predict(voxel)

        return local_exp_pos_map_frame_sorted, exp_pos_sorted, info_sorted
    
    def _check_is_stuck(self,e:int) -> bool:
        """
        Heuristic to check if the agent is stuck:
            1. if the agent is not moving: havn't taken MOVE_FORWARD action for more than 2 x turn steps
            2. if the agent is moving but not making progress: MOVE_FORWARD action is taken for N steps but the agent
                is still in the same grid
        """
        # go to object goal
        if self._state[e] == 2:
            return False
        
        # if total steps is not enough, don't check
        if len(self._pre_actions_taken[e]) < self.pre_action_deque_len:
            return False
        # if the agent is not moving for n consecutive steps        
        if np.all(np.array(self._pre_actions_taken[e]) == DiscreteNavigationAction.MOVE_FORWARD.value) \
            and np.all(self._pre_pose[e][-1] == self._pre_pose[e][0]):
            return True

        
        return False
    
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
        # if self.use_probability_map:
        # we always update the prob map for evaluation
        relevance = torch.zeros(obs.task_observations['semantic_max_val']+1).to(self.device)
        relevance[obs.task_observations['object_goal']] = 1
        relevance[obs.task_observations['start_recep_goal']] = 0.7
        detection_results = {'scores':torch.tensor(obs.task_observations['instance_scores']).unsqueeze(0),
                                'classes':torch.tensor(obs.task_observations['instance_classes']).unsqueeze(0),
                                'masks':torch.tensor(obs.task_observations['masks']).unsqueeze(0),
                                'relevance':relevance}
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
