import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from enum import IntEnum, auto

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
        eval_eps = [f'{e}' for e in eval_eps]
        eps_list = [eps for eps in dataset.episodes if eps.episode_id in eval_eps]
        dataset.episodes = eps_list

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
        agent = OVMMAgent(
            config=self.config,
            device_id=self.gpu_id,
            obs_spaces=self.env.observation_space,
            action_spaces=self.env.action_space,
            collect_data=self.args.collect_data,
        )
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
        start_time = time.time()
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
        while ep_idx < env.number_of_episodes:
            
            current_episodes_info = [self.env.current_episode()]
            print(f'Current pose: {ob.gps*100}, theta: {ob.compass*180/np.pi}')
            action, agent_info, _ = agent.act(ob)
            # print(f'Entropy: {agent_info["entropy"]}, change: {agent_info["entropy"] - pre_entropy}')
            # pre_entropy = agent_info["entropy"]
            
            if not self.args.no_render:

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
                user_action = self.viewer.imshow(
                    draw_ob, delay=0 if not self.args.no_interactive else 2
                )

            if not self.args.no_interactive and user_action is not None:
                if user_action['info'] == "plan_high":
                    agent.force_update_high_goal(0)
                action = user_action['action']

            outputs = env.apply_action(action, agent_info)
            ob, done, info = outputs

            if done:
                print(f"Episode {ep_idx} finished.")
                ob = env.reset()
                ep_idx += 1
                 # update_detic_perception_vocab(ob, detic_perception)
                agent.reset_vectorized_for_env(
                    0, self.env.current_episode()
                )
                visualizer.reset()
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
        default=3,
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
        "--collect_data",
        help="wheter to collect data for training",
        action="store_true",
        default=False,
    )


    print("Arguments:")
    args = parser.parse_args()
    # args.collect_data = True
    print(json.dumps(vars(args), indent=4))
    print("-" * 100)

    print("Configs:")
    
    overrides = []
    config = get_habitat_config(args.habitat_config_path, overrides=overrides)
    baseline_config = OmegaConf.load(args.baseline_config_path)
    config = DictConfig({**config, **baseline_config})
    evaluator = InteractiveEvaluator(config,args.gpu_id,args=args)
    print("-" * 100)
    evaluator.eval(
        num_episodes_per_env=config.EVAL_VECTORIZED.num_episodes_per_env,
    )






