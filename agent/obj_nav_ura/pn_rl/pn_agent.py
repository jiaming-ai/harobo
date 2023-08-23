# adapted from habitat-baseline

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Optional
from home_robot.core.interfaces import DiscreteNavigationAction
import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from omegaconf import DictConfig, OmegaConf

import habitat
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from .resnet_policy import PointNavResNetPolicy
from habitat_baselines.utils.common import batch_obs
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import torchvision.transforms as T
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
from .planning.rrt_star import RRTStar


@dataclass
class PPOAgentConfig:
    INPUT_TYPE: str = "depth"
    MODEL_PATH: str = "data/checkpoints/pn/gibson-4plus-mp3d-train-val-test-resnet50.pth"
    RESOLUTION: int = 256
    HIDDEN_SIZE: int = 512
    RANDOM_SEED: int = 7
    PTH_GPU_ID: int = 0
    GOAL_SENSOR_UUID: str = "pointgoal_with_gps_compass"

@dataclass
class PolicyConfig:
    use_augmentations: bool = False
    action_distribution_type: str = "categorical"


def get_default_config() -> DictConfig:
    return OmegaConf.create(PPOAgentConfig())  # type: ignore[call-overload]


class PPOAgent(Agent):
    def __init__(self, config: DictConfig) -> None:
        spaces = {
            get_default_config().GOAL_SENSOR_UUID: Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(config.RESOLUTION, config.RESOLUTION, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.RESOLUTION, config.RESOLUTION, 3),
                dtype=np.uint8,
            )
        observation_spaces = SpaceDict(spaces)

        action_spaces = Discrete(4)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.HIDDEN_SIZE

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            normalize_visual_inputs="rgb" in spaces,
            backbone="resnet50",
            rnn_type="LSTM",
            num_recurrent_layers=2,
            policy_config=OmegaConf.create(PolicyConfig()),
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations: Observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        return {"action": actions[0][0].item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="rgb",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--task-config",
        type=str,
        default="habitat-lab/habitat/config/task/pointnav.yaml",
    )
    args = parser.parse_args()

    agent_config = get_default_config()
    agent_config.INPUT_TYPE = args.input_type
    if args.model_path is not None:
        agent_config.MODEL_PATH = args.model_path

    agent = PPOAgent(agent_config)
    benchmark = habitat.Benchmark(config_paths=args.task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
    pass




def cartesian_to_polar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def compute_pointgoal(source_position, source_rotation, goal_position):

    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    rho, phi = cartesian_to_polar(
        -direction_vector_agent[2], direction_vector_agent[0]
    )
    return np.array([rho, -phi], dtype=np.float32)
      

########
import torch.nn as nn


# ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]

class PNAgent():
    def __init__(self, device):
        super().__init__()
        self.device = device

        model_path ='data/checkpoints/pn/gibson-4plus-mp3d-train-val-test-resnet50.pth'

        spaces = {
            'pointgoal_with_gps_compass': Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }
        spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(256, 256, 1),
                dtype=np.float32,
            )

        observation_space = SpaceDict(spaces)
        action_space = Discrete(4)

        self.actor_critic = PointNavResNetPolicy(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=512,
            normalize_visual_inputs="rgb" in spaces,
            backbone="resnet50",
            rnn_type="LSTM",
            num_recurrent_layers=2,
            policy_config=OmegaConf.create(PolicyConfig()),
        )
        self.actor_critic.to(self.device)

        ckpt = torch.load(model_path, map_location=self.device)
        #  Filter only actor_critic weights
        self.actor_critic.load_state_dict(
            {  # type: ignore
                k[len("actor_critic.") :]: v
                for k, v in ckpt["state_dict"].items()
                if "actor_critic" in k
            }
        )

       
        self.actor_critic.eval()

        self.hidden_state = torch.zeros(1, self.actor_critic.net.num_recurrent_layers, 512, device=self.device)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

        self.t = 0

    def _process_inputs(self, depth, goal_pose, cur_pose):

        rot_mat = torch.tensor([[torch.cos(cur_pose[2]), -torch.sin(cur_pose[2])],
                                [torch.sin(cur_pose[2]), torch.cos(cur_pose[2])]], dtype=torch.float32 ) # 2,2
        dir_vec = goal_pose-cur_pose[:2] # 2
        dir_vec_agent = rot_mat.T @ dir_vec # 2

        phi = torch.atan2(dir_vec_agent[1],dir_vec_agent[0])
        rho = torch.sqrt(dir_vec[0]**2+dir_vec[1]**2)
        
        # planning_goal = goal_pose
        # planning_pose = cur_pose[:2]
        # sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        # rho = torch.sqrt(sq.float())
        # phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        # phi = (phi - cur_pose[2]) % (2 * np.pi)


        # agent_state = self.sim.get_agent_state()
        # agent_position = agent_state.position
        # rotation_world_agent = agent_state.rotation

        # pg = compute_pointgoal(
        #     agent_position, rotation_world_agent, goal_pose.to_numpy()
        # )
    
        
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32, device=self.device) # 2

        # depth = depth / 10.0 # normalize
        depth = depth[:,:480,:] # crop the bottom part, size 480,480,1
        # depth = depth[:,-480:,:] # crop the top part, size 480,480,1
        depth = T.Resize(size=(256,256))(depth) # resize to 256,256,1
        batch = {'depth': depth.unsqueeze(0).to(self.device), # Bs, C, H, W
                 'pointgoal_with_gps_compass': point_goal_with_gps_compass.unsqueeze(0)} # Bs, 2
        
        return batch

    def _get_action(self, action_key):
        if action_key == 0:
            return DiscreteNavigationAction.STOP
        elif action_key == 1:
            return DiscreteNavigationAction.MOVE_FORWARD
        elif action_key == 2:
            return DiscreteNavigationAction.TURN_LEFT
        elif action_key == 3:
            return DiscreteNavigationAction.TURN_RIGHT
        else:
            raise ValueError("Invalid action key")
    def plan(self, depth, goal_pose, cur_pose):
        """
        args:
            depth: tensor of shape (1, 256, 256), unnormalized depth
            goal_pose: tensor of shape (2,), of (x, y) in world coordinates
            cur_pose: tensor of shape (3,), of (x, y, theta) in world coordinates
        """
        
        batch = self._process_inputs(depth, goal_pose,cur_pose)
        if self.t ==0:
            not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        else:
            not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            _, actions, _, self.hidden_state = self.actor_critic.act(batch,
                                                                    self.hidden_state.to(self.device),
                                                                    self.prev_actions.to(self.device),
                                                                    not_done_masks,
                                                                    deterministic=False)
        self.prev_actions = torch.clone(actions)
        self.t += 1
        
        return self._get_action(actions.item())

    def reset(self):
        self.hidden_state = torch.zeros_like(self.hidden_state)
        self.prev_actions = torch.zeros_like(self.prev_actions)
        self.t = 0
    
    def get_rrt_goal(self, pose_coords, goal, grid, ensemble, prev_path):
        probability_map, indexes = torch.max(grid,dim=1)
        probability_map = probability_map[0]
        indexes = indexes[0]
        binarymap = (indexes == 1)
        start = [int(pose_coords[0][0][1]), int(pose_coords[0][0][0])]
        finish = [int(goal[0][0][1]), int(goal[0][0][0])]
        rrt_star = RRTStar(start=start, 
                           obstacle_list=None, 
                           goal=finish, 
                           rand_area=[0,binarymap.shape[0]], 
                           max_iter=self.options.rrt_max_iters,
                           expand_dis=self.options.expand_dis,
                           goal_sample_rate=self.options.goal_sample_rate,
                           connect_circle_dist=self.options.connect_circle_dist,
                           occupancy_map=binarymap)
        best_path = None
        
        path_dict = {'paths':[], 'value':[]} # visualizing all the paths
        if self.options.exploration:
            paths = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line, exploration=self.options.exploration, horizon=self.options.reach_horizon)
            ## evaluate each path on the exploration objective
            path_sum_var = self.eval_path_expl(ensemble, paths)
            path_dict['paths'] = paths
            path_dict['value'] = path_sum_var

            best_path_var = 0 # we need to select the path with maximum overall uncertainty
            for i in range(len(paths)):
                if path_sum_var[i] > best_path_var:
                    best_path_var = path_sum_var[i]
                    best_path = paths[i]

        else:
            best_path_reachability = float('inf')        
            for i in range(self.options.rrt_num_path):
                path = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line)
                if path:
                    if self.options.rrt_path_metric == "reachability":
                        reachability = self.eval_path(ensemble, path, prev_path)
                    elif self.options.rrt_path_metric == "shortest":
                        reachability = len(path)
                    path_dict['paths'].append(path)
                    path_dict['value'].append(reachability)
                    
                    if reachability < best_path_reachability:
                        best_path_reachability = reachability
                        best_path = path

        if best_path:
            best_path.reverse()
            last_node = min(len(best_path)-1, self.options.reach_horizon)
            return torch.tensor([[[int(best_path[last_node][1]), int(best_path[last_node][0])]]]).cuda(), best_path, path_dict
        return None, None, None