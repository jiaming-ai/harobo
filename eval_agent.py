import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from enum import IntEnum, auto

from home_robot.core.interfaces import DiscreteNavigationAction, Observations, ContinuousNavigationAction
import numpy as np
from omegaconf import DictConfig, OmegaConf
from agent.ovmm.ovmm import OVMMAgent
from utils.ovmm_env_visualizer import Visualizer
HOME_ROBOT_BASE_DIR = str(Path(__file__).resolve().parent.parent / "home-robot") + "/"

sys.path.insert(
    0,
    HOME_ROBOT_BASE_DIR + "src/home_robot"
)
sys.path.insert(
    0,
    HOME_ROBOT_BASE_DIR + "src/home_robot_sim"
)
import cv2
from habitat import make_dataset
from habitat.core.environments import get_env_class
from habitat.core.vector_env import VectorEnv
from habitat.utils.gym_definitions import _get_env_name
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer

# from home_robot.agent.ovmm_agent.ovmm_agent import OpenVocabManipAgent
from home_robot_sim.env.habitat_ovmm_env.habitat_ovmm_env import (
    HabitatOpenVocabManipEnv,
)
from typing import Optional, Tuple

from habitat_baselines.config.default import get_config as get_habitat_config
from omegaconf import DictConfig
from utils.viewer import OpenCVViewer
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
import torch
import random

# NON_SCALAR_METRICS = {"top_down_map", "collisions.is_collision"}
# METRICS = ['OVMMDistToPickGoal', # distance to pick goal
#            'ovmm_nav_to_pick_succ' # success of navigation to pick goal
#            ]

# def extract_scalars_from_info(cls, info):
#     result = {}
#     for k, v in info.items():
#         if not isinstance(k, str) or k in NON_SCALAR_METRICS:
#             continue

#         if isinstance(v, dict):
#             result.update(
#                 {
#                     k + "." + subk: subv
#                     for subk, subv in cls._extract_scalars_from_info(
#                         v
#                     ).items()
#                     if isinstance(subk, str)
#                     and k + "." + subk not in NON_SCALAR_METRICS
#                 }
#             )
#         # Things that are scalar-like will have an np.size of 1.
#         # Strings also have an np.size of 1, so explicitly ban those
#         elif np.size(v) == 1 and not isinstance(v, str):
#             result[k] = float(v)

#     return result

def get_total_navigable_area(env):
    """
    Returns the total navigable area in the environment
    """
    sim = env.habitat_env.env._env.habitat_env.sim
    pf = sim.pathfinder
    return pf.navigable_area

def create_ovmm_env_fn(config,args):
    """Create habitat environment using configsand wrap HabitatOpenVocabManipEnv around it. This function is used by VectorEnv for creating the individual environments"""
    
    if args.collect_data:
        splits = ['train','val','test']
        OmegaConf.set_readonly(config, False)
        config.habitat.dataset.split = 'train'
        config.habitat.task.episode_init=False
        OmegaConf.set_readonly(config, True)

        habitat_config = config.habitat
        dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)
    
        # we select a subset of episodes to generate the dataset
        eps_select = {}
        eps_list = []
        skip = 12
        eps_per_scene = 12
        for eps in dataset.episodes:
            scene_id = eps.scene_id
            if scene_id not in eps_select:
                eps_select[scene_id] = 0
            eps_select[scene_id] += 1
            if eps_select[scene_id] < skip:
                continue
            if eps_select[scene_id] < skip + eps_per_scene:
                eps_list.append(eps)
        dataset.episodes = eps_list

    else:
        habitat_config = config.habitat
        dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)

    if args.eval_eps is not None:
        eval_eps = [f'{e}' for e in args.eval_eps]
        eps_list = [eps for eps in dataset.episodes if eps.episode_id in eval_eps]
        dataset.episodes = eps_list

    if args.eval_eps_total_num is not None:
        dataset.episodes = dataset.episodes[:args.eval_eps_total_num]

    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env


class InteractiveEvaluator():
    """class for interactive evaluation of OpenVocabManipAgent on an episode dataset"""
    def __init__(self, config, gpu_id: int = 0, args=None):
        self.visualize = config.VISUALIZE or config.PRINT_IMAGES

        episode_ids_range = config.habitat.dataset.episode_indices_range
        if episode_ids_range is not None:
            config.EXP_NAME = os.path.join(
                config.EXP_NAME, f"{episode_ids_range[0]}_{episode_ids_range[1]}"
            )
        OmegaConf.set_readonly(config, True)

        self.config = config
        self.results_dir = os.path.join(
            self.config.DUMP_LOCATION, "results", self.config.EXP_NAME
        )
        self.videos_dir = self.config.habitat_baselines.video_dir
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        self.agent = None
        self.env = None
        self.gpu_id = gpu_id
        self.args = args
        self.viewer = OpenCVViewer() if not args.no_render else None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.close()


    def eval(self, num_episodes_per_env=10):
     
        self.env = create_ovmm_env_fn(self.config,self.args)
        visualize=self.args.save_video or (not self.args.no_render) or (not self.args.no_interactive)
        print(f'Env created')
        agent = OVMMAgent(
            config=self.config,
            device_id=self.gpu_id,
            obs_spaces=self.env.observation_space,
            action_spaces=self.env.action_space,
            collect_data=self.args.collect_data,
            eval_rl_nav=(config.AGENT.SKILLS.NAV_TO_OBJ.type == "rl"),
            use_FBE_policy=self.args.eval_policy == 'fbe',
            visualize=visualize,
        )

        print(f'Agent created')
        self.play(
            agent,
            self.env,
            num_episodes_per_env=num_episodes_per_env,
            episode_keys=None,
        )

    def write_results(self, episode_metrics):
        aggregated_metrics = defaultdict(list)
        metrics = set(
            [
                k
                for metrics_per_episode in episode_metrics.values()
                for k in metrics_per_episode
                if k != "goal_name"
            ]
        )
        for v in episode_metrics.values():
            for k in metrics:
                if k in v:
                    aggregated_metrics[f"{k}/total"].append(v[k])

        aggregated_metrics = dict(
            sorted(
                {
                    k2: v2
                    for k1, v1 in aggregated_metrics.items()
                    for k2, v2 in {
                        f"{k1}/mean": np.mean(v1),
                        f"{k1}/min": np.min(v1),
                        f"{k1}/max": np.max(v1),
                    }.items()
                }.items()
            )
        )
        with open(f"{self.results_dir}/aggregated_results.json", "w") as f:
            json.dump(aggregated_metrics, f, indent=4)
        with open(f"{self.results_dir}/episode_results.json", "w") as f:
            json.dump(episode_metrics, f, indent=4)


    def play(
        self,
        agent: OVMMAgent,
        env: HabitatOpenVocabManipEnv,
        num_episodes_per_env=None,
        episode_keys=None,
    ):
        # The stopping condition is either specified through
        # num_episodes_per_env (stop after each environment
        # finishes a certain number of episodes)
        print(f"Running eval on {env.number_of_episodes} episodes")
        # detic_perception = OvmmPerception(self.config,self.gpu_id)

        episode_metrics = {}
        print(f'Resetting...')
        ob = env.reset()
        # update_detic_perception_vocab(ob, detic_perception)

        visualizer = Visualizer(self.config,self.env._dataset)
        visualizer.reset()
        agent.reset_vectorized([self.env.current_episode()])
        agent_info = None
        print('*'*20)
        print(f'Goal: {ob.task_observations["goal_name"]}')
        pre_entropy = 0
        ep_idx = 0
        coverage_log_interval = 10
        ########################################
        # init evaluation metrics
        # only used for object navigation task
        ########################################
        visualize=self.args.save_video or (not self.args.no_render) or (not self.args.no_interactive)
        results = []
        recorder = Recording()
        result_dir = f'datadump/exp_results/{self.args.exp_name}/'
        os.makedirs(result_dir, exist_ok=True)

        want_terminate = False
        forward_steps = 0
        init_dts = env.habitat_env.env._env.habitat_env.get_metrics()['ovmm_dist_to_pick_goal']
        exp_coverage_list = []
        checking_area_list = []
        entropy_list = []
        close_coverage_list = []
        eps_step = 0
        
        while ep_idx < env.number_of_episodes:
            eps_step += 1
            
            print(f'Current pose: {ob.gps*100}, theta: {ob.compass*180/np.pi}')
            action, agent_info, _ = agent.act(ob)
            # print(f'Entropy: {agent_info["entropy"]}, change: {agent_info["entropy"] - pre_entropy}')
            # pre_entropy = agent_info["entropy"]
            print(f'exp_area: {agent_info["exp_coverage"]}, checking_area: {agent_info["checking_area"]}')

            exp_coverage_list.append(agent_info["exp_coverage"])
            checking_area_list.append(agent_info["checking_area"]+checking_area_list[-1] \
                if len(checking_area_list) > 0 else agent_info["checking_area"])
            entropy_list.append(agent_info["entropy"])
            close_coverage_list.append(agent_info["close_coverage"])

            if visualize:
                draw_ob = ob.rgb

                # # visualize GT semantic map
                # gt_semantic = env.visualizer.get_semantic_vis(ob.semantic,ob.rgb)
                # draw_ob = np.concatenate([draw_ob,gt_semantic], axis=1)

                #  visualize detected instances
                if 'semantic_frame' in ob.task_observations:
                    draw_ob = np.concatenate([draw_ob, ob.task_observations['semantic_frame']], axis=1)
                
                # # visualize objectness
                # if 'objectiveness_visualization' in ob.task_observations:
                #     draw_ob = np.concatenate([draw_ob, ob.task_observations['objectiveness_visualization']], axis=1)
                
                # visualize semantic map
                vis = visualizer.visualize(**agent_info)
                semantic_map_vis = vis['semantic_map']
                semantic_map_vis = cv2.resize(
                    semantic_map_vis,
                    (640, 640),
                    interpolation=cv2.INTER_NEAREST,
                )
                draw_ob = np.concatenate([draw_ob, semantic_map_vis], axis=1)

                # visualize probabilistic map
                if "probabilistic_map" in agent_info and agent_info['probabilistic_map'] is not None:
                    prob_map = agent_info['probabilistic_map']
                    prob_map = np.flipud(prob_map)
                    prob_map = cv2.resize(
                        prob_map,
                        (640, 640),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    prob_map = (prob_map * 255).astype(np.uint8)
                    prob_map = cv2.cvtColor(prob_map, cv2.COLOR_GRAY2BGR)
                    draw_ob = np.concatenate([draw_ob, prob_map], axis=1)
                
                # visualize info_gain map
                if "info_map" in agent_info and agent_info['info_map'] is not None:
                    info_map = agent_info['info_map']
                    info_map = np.flipud(info_map)
                    info_map = cv2.resize(
                        info_map,
                        (640, 640),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    info_map = (info_map * 255).astype(np.uint8)
                    if info_map.ndim == 2:
                        info_map = cv2.cvtColor(info_map, cv2.COLOR_GRAY2BGR)
                    draw_ob = np.concatenate([draw_ob, info_map], axis=1)
                    # draw

            if self.args.save_video:
                recorder.add_frame(draw_ob)

            if not self.args.no_render:
                user_action = self.viewer.imshow(
                    draw_ob, delay=0 if not self.args.no_interactive else 2
                )

            if not self.args.no_interactive and user_action is not None:
                if user_action['info'] == "plan_high":
                    agent.force_update_high_goal(0)
                action = user_action['action']

            outputs = env.apply_action(action, agent_info)
            ob, done, info = outputs

            if agent_info["skill_done"] != '':
                want_terminate = True
                done = True
            elif action == DiscreteNavigationAction.MOVE_FORWARD:
                forward_steps += 1
            print(action)
            
            
            if done:
                print(f"Episode {ep_idx} finished.")
                current_episodes_info = self.env.current_episode()
                
                # save evaluation results
                total_nav_area = get_total_navigable_area(env) # in m2
                eps_result = {
                    'episode_id': current_episodes_info.episode_id,
                    'scene_id': current_episodes_info.scene_id,
                    'success': info['ovmm_nav_to_pick_succ'],
                    'distance_to_goal': info['ovmm_dist_to_pick_goal'],
                    'travelled_distance': forward_steps * self.config.ENVIRONMENT.forward,
                    'steps': info['num_steps'],
                    'want_terminate': want_terminate,
                    'goal_object': ob.task_observations["goal_name"],
                    'spl': init_dts / max(forward_steps * self.config.ENVIRONMENT.forward, init_dts),
                    'total_nav_area': total_nav_area,
                    'exp_coverage': exp_coverage_list,
                    'checking_area': checking_area_list,
                    'entropy': entropy_list,
                    'close_coverage': close_coverage_list,
                }
                results.append(eps_result)
                fname = f'{ob.task_observations["goal_name"]}_{info["ovmm_nav_to_pick_succ"]}'
                with open(f'{result_dir}/{fname}.json', 'w') as f:
                    json.dump(eps_result, f)
                if self.args.save_video:
                    recorder.save_video(fname,result_dir)
                
                # reset env and agent
                ob = env.reset()
                agent.reset_vectorized_for_env(
                    0, self.env.current_episode()
                )
                visualizer.reset()
                ep_idx += 1

                # reset eval metrics
                want_terminate = False
                forward_steps = 0
                init_dts = env.habitat_env.env._env.habitat_env.get_metrics()['ovmm_dist_to_pick_goal']
                exp_coverage_list = []
                checking_area_list = []
                entropy_list = []
                close_coverage_list = []
                eps_step = 0

                print("*"*20)
                print(f'Goal: {ob.task_observations["goal_name"]}')

        env.close()
        self.write_results(episode_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--habitat_config_path",
        type=str,
        default="ovmm/ovmm_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--baseline_config_path",
        type=str,
        default="configs/agent/hssd_eval.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="GPU id to use for evaluation",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Whether to render the environment or not",
        default=False,
    )
    parser.add_argument(
        "--no_interactive",
        action="store_true",
        help="Whether to render the environment or not",
        default=False,
    )
    parser.add_argument(
        "--eval_eps",
        help="evaluate a subset of episodes",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--eval_eps_total_num",
        help="evaluate a subset of episodes",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--collect_data",
        help="wheter to collect data for training",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exp_name",
        help="experiment name",
        type=str,
        default='debug',
    )
    
    parser.add_argument(
        "--save_video",
        help="Save video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--eval_policy",
        help="policy to evaluate: fbe | rl | ur",
        type=str,
        default='ur',
    )
    parser.add_argument(
        "--seed",
        help="random seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gt_semantic",
        help="whether to use ground truth semantic map",
        action="store_true",
        default=False,    
    )
    parser.add_argument(
        "--no_use_prob_map",
        help="whether to use probability map",
        action="store_true",
        default=False,
    )
    
    print("Arguments:")
    args = parser.parse_args()
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

  
    print("Configs:")
    

    overrides = []
    config = get_habitat_config(args.habitat_config_path, overrides=overrides)
    baseline_config = OmegaConf.load(args.baseline_config_path)
    config = DictConfig({**config, **baseline_config})
    evaluator = InteractiveEvaluator(config,args.gpu_id,args=args)
    print("-" * 100)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    OmegaConf.set_readonly(config, False)
    config.habitat.seed = seed
    if args.eval_policy == 'rl':
        config.AGENT.SKILLS.NAV_TO_OBJ.type = "rl"
    elif args.eval_policy == 'fbe':
        config.AGENT.SKILLS.NAV_TO_OBJ.type = "heuristic"
    elif args.eval_policy == 'ur':
        config.AGENT.SKILLS.NAV_TO_OBJ.type = "heuristic"
    else:
        raise ValueError(f'Unknown policy type: {args.eval_policy}')
    config.GROUND_TRUTH_SEMANTICS = 1 if args.gt_semantic else 0

    # if args.eval_policy == 'ur' and not args.no_use_prob_map and not args.gt_semantic:
    #     config.AGENT.SEMANTIC_MAP.use_probability_map = True
    # else:
    #     config.AGENT.SEMANTIC_MAP.use_probability_map = False
    visualize = args.save_video or (not args.no_render) or (not args.no_interactive)
    
    OmegaConf.set_readonly(config, True)
    
    evaluator.eval(
        num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env,
    )






