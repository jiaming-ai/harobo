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


# base_dir = osp.abspath(osp.join(osp.dirname(__file__), ".."))



def create_ovmm_env_fn(config,split='train'):
    """Create habitat environment using configsand wrap HabitatOpenVocabManipEnv around it. This function is used by VectorEnv for creating the individual environments"""

    
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = split
    config.habitat.task.episode_init=False
    OmegaConf.set_readonly(config, True)

    habitat_config = config.habitat
    dataset = make_dataset(habitat_config.dataset.type, config=habitat_config.dataset)

    # we select a subset of episodes to generate the dataset
    eps_select = {}
    eps_list = []
    skip = 0
    eps_per_scene = 1
    for eps in dataset.episodes:
        scene_id = eps.scene_id
        if scene_id not in eps_select:
            eps_select[scene_id] = 0
        if eps_select[scene_id] < skip:
            continue
        if eps_select[scene_id] < skip + eps_per_scene:
            eps_list.append(eps)
        eps_select[scene_id] += 1

    dataset.episodes = eps_list

  
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    habitat_env = env_class(config=habitat_config, dataset=dataset)
    habitat_env.seed(habitat_config.seed)
    env = HabitatOpenVocabManipEnv(habitat_env, config, dataset=dataset)
    return env


def generate_maps(split='train'):
    # optionally generate human-readable map images
    os.makedirs('data/maps', exist_ok=True)
    config = get_habitat_config('ovmm/ovmm_eval.yaml', overrides=[])
    baseline_config = OmegaConf.load('configs/agent/hssd_eval.yaml')
    config = DictConfig({**config, **baseline_config})
    env = create_ovmm_env_fn(config,split)
    maps = {}

    while True:
        env.reset()
        scene_id = env.get_current_episode().scene_id
        scene_id = scene_id.split('/')[-1].split('.')[0]

        sim = env.habitat_env.env._env.habitat_env.sim
        pf = sim.pathfinder

        # random sample navigable points, and use the most common y value
        y = []
        for i in range(100):
            pos = pf.get_random_navigable_point()
            y.append(pos[1])
        y = np.array(y)
        v,c = np.unique(y, return_counts=True)
        y = v[np.argmax(c)]
        # 
            
        nav_map = pf.get_topdown_view(0.05, y)
        # plot_image(nav_map)

        if scene_id not in maps:
            maps[scene_id] = nav_map
            print('saving map for scene', scene_id)
            np.save('data/maps/' + scene_id, nav_map)
        else:
            break


for s in ['train','val']:
    generate_maps(s)