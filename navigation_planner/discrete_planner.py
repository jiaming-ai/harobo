# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import shutil
import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology

import home_robot.utils.pose as pu
from home_robot.core.interfaces import (
    ContinuousNavigationAction,
    DiscreteNavigationAction,
)
from home_robot.utils.geometry import xyt_global_to_base

from .fmm_planner import FMMPlanner
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
    save_img_tensor)
CM_TO_METERS = 0.01


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]


class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        visualize: bool,
        print_images: bool,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,
        discrete_actions: bool = True,
        continuous_angle_tolerance: float = 5.0,
    ):
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            visualize: if True, render planner internals for debugging
            print_images: if True, save visualization as images
        """
        self.discrete_actions = discrete_actions
        self.visualize = visualize
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None
        self.collision_map = None
        self.visited_map = None
        self.col_width = None
        self.last_pose = None
        self.curr_pose = None
        self.last_action = None
        self.timestep = 0
        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.cur_min_goal_distance_cm = min_goal_distance_cm
        self.init_min_goal_distance_cm = min_goal_distance_cm
        self.dd = None

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency
        
        self.stuck_count = 0
        self.last_lmb = None
        self.dd_planner = None
        self.navigable_goal_map = None
        self.max_goal_distance_cm = 80 
        self.last_rid = None
        
    def reset(self):
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.col_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.goal_dilation_selem = skimage.morphology.disk(
            self.goal_dilation_selem_radius
        )

        self.global_goal_pose = None
        self.cur_min_goal_distance_cm = self.init_min_goal_distance_cm
        self.stuck_count = 0
        self.last_lmb = [240,720,240,720]
        self.dd_planner = None
        self.navigable_goal_map = None
        self.high_goal_tolerance = 4 # 25 cm
        self.max_goal_distance_cm = 80 # 80 cm
        self.last_rid = None
        
    def set_vis_dir(self, scene_id: str, episode_id: str):
        self.vis_dir = os.path.join(self.default_vis_dir, f"{scene_id}_{episode_id}")
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def disable_print_images(self):
        self.print_images = False

    def relax_goal_tolerance(self,rid=None):
        if self.last_rid is None or self.last_rid != rid:
            # self.goal_tolerance += 2
            self.collision_map *= 0
            if self.max_goal_distance_cm < 110:
                self.max_goal_distance_cm += 10 
            print(f"Re-planning for object goal, increased goal tolerance to {self.goal_tolerance}")

            # Reduce obstacle dilation
            if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(self.curr_obs_dilation_selem_radius)
                print(f"Re-planning for object goal, reduced obs dilation to {self.curr_obs_dilation_selem_radius}")
            self.last_rid = rid
        
    def plan(
        self,
        global_obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
        debug: bool = True,
        timestep: int = None,
        global_goal_pose: np.ndarray = None,
        map_changed: bool = False,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray]:
        """Plan a low-level action.

        1. to speed up, we only recompute the distance map when the map has changed & goal has changed
        2. when found_goal is specified, we try to get as close as possible to the goal by reducing the goal threshold,
        and orient towards the goal object
        3. we seems don't need closesed goal map, so we remove it
        4. use continuous action to move towards faster (not sure if it's better)?

        Args:
            obstacle_map: (N, N) global obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            replan_hgoal: the high goal is not reachable, replan a new one and mark the old one as not reachable
            reach_hgoal: the high goal is reachaed
            
        """
        replan_hgoal = False
        end_episode = False
        reach_hgoal = False
        getting_out = False
        # Reset timestep using argument; useful when there are timesteps where the discrete planner is not invoked
        if timestep is not None:
            self.timestep = timestep

        # Update the global goal pose
        if self.global_goal_pose is None or (not np.allclose(global_goal_pose, self.global_goal_pose)):
            new_hgoal = True
            self.cur_min_goal_distance_cm = self.init_min_goal_distance_cm
            self.global_goal_pose = global_goal_pose
            self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        else:
            new_hgoal = False

        self.last_pose = self.curr_pose

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        self.curr_pose = [start_x, start_y, start_o]

        # use same gx1, gx2, gy1, gy2 as the previous one, if the map has not changed
        # we only update navigable goal map and dd when the map or goal has changed
        update_dd=map_changed or new_hgoal
        if update_dd:
            self.last_lmb = [gx1, gx2, gy1, gy2]
            self.navigable_goal_map = None
            # self.dd_planner = None
        
        # if not updating dd, use the last lmb
        else:
            gx1, gx2, gy1, gy2 = self.last_lmb
        
        planning_window = [gx1, gx2, gy1, gy2]
        obstacle_map = np.rint(global_obstacle_map[gx1:gx2, gy1:gy2])
        
        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        start = pu.threshold_poses(start, obstacle_map.shape)
        start = np.array(start)

        if debug:
            print()
            print("--- Planning ---")
            print("Found goal:", found_goal)
            print("Goal points provided:", np.any(goal_map > 0))

        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1
        
        # Check collisions if we have just moved and are uncertain
        # this is to update the collision map
        is_collide = False
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            is_collide = self._check_collision()
        
            # check if we get stuck by comparing the current pose with the last pose
            # collision is only checked when the agent is moving
            # stuck can  happen even when the agent is not moving
            # this is to get the agent out of stuck
            if np.allclose(self.last_pose, self.curr_pose):
                self.stuck_count += 1

            # clear the stuck count if the agent is moving
            else:
                self.stuck_count = 0

        # low level stuck get out
        # if the agent can't avoid the obstacle by adding collision map
        # we try to get out of stuck with some random actions
        if self.stuck_count > 2:
            print("Warning: agent is stuck, try to get out of stuck...")
            short_term_goal = self.getout_stuck(obstacle_map, start)
            closest_goal_map = None
            no_path_found, stop = False, False
            closest_goal_pt = None
            dilated_obstacles = None
            reach_hgoal = False
            getting_out = True
          

        # if not stuck, move towards the goal
        else:
            (
                short_term_goal,
                closest_goal_map,
                no_path_found,
                stop,
                closest_goal_pt,
                dilated_obstacles,
                hgoal_dist,
            ) = self._get_short_term_goal(
                obstacle_map,
                np.copy(goal_map),
                start,
                planning_window,
                is_ur_goal=not found_goal,
                update_goal_map=update_dd,
            )
            if not found_goal:
                reach_hgoal = hgoal_dist < self.high_goal_tolerance
            
            # TODO: handle crash?
            # replan_hgoal = True

        # Short term goal is in cm, start_x and start_y are in m
        if debug:
            print("Current pose:", start)
            print("Short term goal:", short_term_goal)
            print(
                "  - delta =",
                short_term_goal[0] - start[0],
                short_term_goal[1] - start[1],
            )
            dist_to_short_term_goal = np.linalg.norm(
                start - np.array(short_term_goal[:2])
            )
            print(
                "Distance (m):",
                dist_to_short_term_goal * self.map_resolution * CM_TO_METERS,
            )
            print("Replan:", no_path_found)
      
        # # if no path found to hgoal, we should replan a new hgoal
        # if not found_goal and no_path_found:
        #     replan_hgoal = True
        #     print("Cannot go to high goal, replan a new one")

        # for object goal, we relax the planner if we cannot reach the goal
        if no_path_found and not stop:
            # Clean collision map
            self.collision_map *= 0
            # Reduce obstacle dilation
            while self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )
                if debug:
                    print(
                        f"Re-planning for object goal, reduced obs dilation to {self.curr_obs_dilation_selem_radius}"
                    )

                (
                    short_term_goal,
                    closest_goal_map,
                    no_path_found,
                    stop,
                    closest_goal_pt,
                    dilated_obstacles,
                    hgoal_dist,
                ) = self._get_short_term_goal(
                    obstacle_map,
                    np.copy(goal_map),
                    start,
                    planning_window,
                    is_ur_goal=not found_goal,
                )
                if debug:
                    print("--- after replanning ---")
                    print("goal =", short_term_goal)
                    
                if not no_path_found:
                    break

            if no_path_found:
                if found_goal:
                    print('Cannot go to object goal even after replanning. Relax goal tolerance')
                    self.relax_goal_tolerance()
                else:
                    # if we still cannot find a path, we try to get out of stuck by moving to some random navigable area
                    print("Cannot go to ur goal even after replanning. Try some random actions")
                short_term_goal = self.getout_stuck(obstacle_map, start)
                closest_goal_map = None
                no_path_found, stop = False, False
                closest_goal_pt = None
                dilated_obstacles = None
                reach_hgoal = False
                replan_hgoal = True
                getting_out = True

        # Normalize agent angle
        angle_agent = pu.normalize_angle(start_o)

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - start[0], stg_y - start[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = pu.normalize_angle(angle_agent - angle_st_goal)

        # Compute angle to the final goal
        # only need to orient to goal if we found goal
        relative_angle_to_closest_goal = None
        if found_goal:
            xs, ys = goal_map.nonzero()
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
            closest_goal_pt = [xc, yc]
            
            goal_x, goal_y = closest_goal_pt
            angle_goal = math.degrees(math.atan2(goal_x - start[0], goal_y - start[1]))
            relative_angle_to_closest_goal = pu.normalize_angle(angle_agent - angle_goal)

            if debug:
                # Actual metric distance to goal
                distance_to_goal = np.linalg.norm(np.array([goal_x, goal_y]) - start)
                distance_to_goal_cm = distance_to_goal * self.map_resolution
                # Display information
                print("-----------------")
                print("Found reachable goal:", found_goal)
                print("Stop:", stop)
                print("Angle to goal:", relative_angle_to_closest_goal)
                print("Distance to goal", distance_to_goal)
                print(
                    "Distance in cm:",
                    distance_to_goal_cm,
                    ">",
                    self.cur_min_goal_distance_cm,
                )

                m_relative_stg_x, m_relative_stg_y = [
                    CM_TO_METERS * self.map_resolution * d
                    for d in [relative_stg_x, relative_stg_y]
                ]
                print("continuous actions for exploring")
                print("agent angle =", angle_agent)
                print("angle stg goal =", angle_st_goal)
                print("angle final goal =", relative_angle_to_closest_goal)
                print(
                    m_relative_stg_x, m_relative_stg_y, "rel ang =", relative_angle_to_stg
                )
                print("-----------------")

        # if reach ur goal, no need to orient towards the goal
        if not found_goal and stop:
            print("Reached goal. Stop.")
            reach_hgoal = True
            # take some random actions if we reach the ur goal but not the object goal
            action = DiscreteNavigationAction.TURN_LEFT
        else:
            action = self.get_action(
                relative_stg_x,
                relative_stg_y,
                relative_angle_to_stg,
                relative_angle_to_closest_goal,
                start_o,
                found_goal,
                stop,
                debug,
                getting_out,
                
            )
        self.last_action = action

        ret = {
            "action": action,
            "short_term_goal": short_term_goal,
            "closest_goal_map": closest_goal_map,
            "no_path_found": no_path_found,
            "stop": stop,
            "closest_goal_pt": closest_goal_pt,
            "dilated_obstacles": dilated_obstacles,
            "replan_hgoal": replan_hgoal,
            "end_episode": end_episode,
            "reach_hgoal": reach_hgoal,
        }
        return ret

    def getout_stuck(self,obstacle_map, start):
        """
        Get out of stuck by moving to some random navigable area within a distance
        """
        dilated_obstacles = cv2.dilate(obstacle_map, self.obs_dilation_selem, iterations=2)

        agent_rad = self.agent_cell_radius + 4 # incrementally try farther away
        agent_map = np.zeros_like(obstacle_map,dtype=bool)


        while True:
            agent_rad += 1
            agent_map[
                int(start[0]) - agent_rad : int(start[0]) + agent_rad + 1,
                int(start[1]) - agent_rad : int(start[1]) + agent_rad + 1,
            ] = True

            agent_map[dilated_obstacles == 1] = False
            agent_map_eroded = cv2.erode(agent_map.astype(np.float32), self.obs_dilation_selem, iterations=1)

            if np.any(agent_map_eroded):
                possible_goals = np.nonzero(agent_map_eroded)
                print("Try to find a goal this is not close to the obstacle")
            elif np.any(agent_map):
                possible_goals = np.nonzero(agent_map)
                print("Try to find a goal anywhere nearby")
            else:
                print("Can't find a goal, increase the agent radius")
                continue
            idx = np.random.randint(len(possible_goals[0]))
            goal = [possible_goals[0][idx], possible_goals[1][idx]]
        
            return goal
            
    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        found_goal: bool,
        stop: bool,
        debug: bool,
        getting_out: bool = False,
    ):
        """
        Gets discrete/continuous action given short-term goal. Agent orients to closest goal if found_goal=True and stop=True
        args:
            relative_angle_to_closest_goal: this should be the angle to the object goal, only used for object goal
        """
        # Short-term goal -> deterministic local policy
        if not (found_goal and stop):
            if self.discrete_actions:
                if relative_angle_to_stg > self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_stg < -self.turn_angle / 2.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.MOVE_FORWARD
            else:
                # Use the short-term goal to set where we should be heading next
                m_relative_stg_x, m_relative_stg_y = [
                    CM_TO_METERS * self.map_resolution * d
                    for d in [relative_stg_x, relative_stg_y]
                ]
                # turn towards the goal to ensure the map is updated
                # when the short term goal is not in the current view (fov is 42)
                at = 180 if getting_out else 20
                if np.abs(relative_angle_to_stg) > at:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    action = ContinuousNavigationAction([0, 0, -relative_angle_to_stg])
                else:
                    # Must return commands in radians and meters
                    relative_angle_to_stg = math.radians(relative_angle_to_stg)
                    xyt_global = [
                        m_relative_stg_y,
                        m_relative_stg_x,
                        -relative_angle_to_stg,
                    ]

                    xyt_local = xyt_global_to_base(
                        xyt_global, [0, 0, math.radians(start_compass)]
                    )
                    xyt_local[2] = -relative_angle_to_stg  # the original angle was already in base frame
                    action = ContinuousNavigationAction(xyt_local)
        else:
            # Try to orient towards the goal object - or at least any point sampled from the goal
            # object.
            if debug:
                print()
                print("----------------------------")
                print(">>> orient towards the goal:", relative_angle_to_closest_goal)
            if self.discrete_actions:
                if relative_angle_to_closest_goal > 2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_RIGHT
                elif relative_angle_to_closest_goal < -2 * self.turn_angle / 3.0:
                    action = DiscreteNavigationAction.TURN_LEFT
                else:
                    action = DiscreteNavigationAction.STOP
            elif (
                np.abs(relative_angle_to_closest_goal) > self.continuous_angle_tolerance
            ):
                if debug:
                    print("Continuous rotation towards goal point")
                relative_angle_to_closest_goal = math.radians(
                    relative_angle_to_closest_goal
                )
                action = ContinuousNavigationAction(
                    [0, 0, -relative_angle_to_closest_goal]
                )
            else:
                action = DiscreteNavigationAction.STOP
                if debug:
                    print("!!! DONE !!!")

        return action

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: List[int],
        planning_window: List[int],
        visualize=False,
        is_ur_goal=False,
        update_goal_map=True,
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
        """
        goal_distance_map, closest_goal_pt = None, None
        # Dilate obstacles, we need this for visualization,
        # move this inside the if can speed up
        dilated_obstacles = cv2.dilate(obstacle_map, self.obs_dilation_selem, iterations=1)

        gx1, gx2, gy1, gy2 = planning_window

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2] == 1] = 0

        agent_rad = self.agent_cell_radius

        # NOTE: increasing agent radius is not helpful, because it can hardly find path and get out of stuck
        traversible[
            int(start[0]) - agent_rad : int(start[0]) + agent_rad + 1,
            int(start[1]) - agent_rad : int(start[1]) + agent_rad + 1,
        ] = 1
            
        traversible[self.visited_map[gx1:gx2, gy1:gy2] == 1] = 1

        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)

        self.dd_planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            visualize=self.visualize,
            print_images=self.print_images,
            goal_tolerance=self.goal_tolerance,
        )
    
        # THIS IS TO MAKE SURE GOAL IS NOT SWOLLEN BY DILATION
        # we should gradually increase the goal distance, when short term goal is not reachable
        # (or replan is true)
        # we only do this for object goal, not for ur goal
        max_goal_distance_cm = self.max_goal_distance_cm
            
        while self.cur_min_goal_distance_cm <= max_goal_distance_cm:
            # TODO: we can transform the previous goal map to avoid re computing the goal map
            if update_goal_map or self.navigable_goal_map is None:
                navigable_goal_map = self.dd_planner._find_within_distance_to_multi_goal(
                    goal_map,
                    self.cur_min_goal_distance_cm / self.map_resolution,
                    timestep=self.timestep,
                    vis_dir=self.vis_dir,
                )
            
            else:
                navigable_goal_map = self.navigable_goal_map
            # # if no navigable goal, increase the goal distance
            if not navigable_goal_map.any():
                self.cur_min_goal_distance_cm += 10
                continue

            # NOTE: we need to update dd every step to ensure the agent's current location is in the navigable area
            self.dd = self.dd_planner.set_multi_goal(
                navigable_goal_map,
                self.timestep,
                self.dd,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            
            state = [start[0] + 1, start[1] + 1]
            # This is where we create the planner to get the trajectory to this state
            stg_x, stg_y, no_path_found, stop, dist_to_goal = self.dd_planner.get_short_term_goal(
                state, continuous=(not self.discrete_actions)
            )

            # if no path to ur goal, we shuold replan a new hgoal
            # otherwise (object goal), we try again with a larger goal distance
            if (not is_ur_goal) and no_path_found and (not stop):
                    self.cur_min_goal_distance_cm += 10
                    update_goal_map = True
            # only increase the goal distance when going to object goal and not found
            else:
                break
        # no path can be found even with the largest goal distance
        if self.cur_min_goal_distance_cm > max_goal_distance_cm:
            no_path_found = True
            print("Cannot find a path to the goal even with the largest goal distance")
            return None, None, no_path_found, None, None, None, None

        
        self.navigable_goal_map = navigable_goal_map
        # if not is_ur_goal:
        #     goal_distance_map, closest_goal_pt = self.get_closest_goal(goal_map, start)

        self.timestep += 1
        # hgoal_dist = self.dd[1:-1,1:-1][agent_map].min()

        stg_x, stg_y = stg_x - 1, stg_y - 1
        short_term_goal = int(stg_x), int(stg_y)

        if visualize:
            print("Start visualizing")
            plt.figure(1)
            plt.subplot(131)
            _navigable_goal_map = self.dd_planner.goal_map.copy()
            # _navigable_goal_map[int(stg_y), int(stg_x)] = 1
            plt.imshow(np.flipud(_navigable_goal_map))
            plt.plot(481-stg_x, 1+stg_y, "bx")
            plt.plot(481-start[0], 1+start[1], "rx")
            plt.subplot(132)
            plt.imshow(np.flipud(self.dd_planner.fmm_dist))
            plt.subplot(133)
            plt.imshow(np.flipud(self.dd_planner.traversible))
            plt.show()
            print("Done visualizing.")

        return (
            short_term_goal,
            goal_distance_map,
            no_path_found,
            stop,
            closest_goal_pt,
            dilated_obstacles,
            dist_to_goal,
        )

    def get_closest_traversible_goal(
        self, traversible, goal_map, start, dilated_goal_map=None
    ):
        """Old version of the get_closest_goal function, which takes into account the distance along geometry to a goal object. This will tell us the closest point on the goal map, both for visualization and for orienting towards it to grasp. Uses traversible to sort this out."""

        # NOTE: this is the old version - before adding goal dilation
        # vis_planner = FMMPlanner(traversible)
        # TODO How to do this without the overhead of creating another FMM planner?
        traversible_ = traversible.copy()
        if dilated_goal_map is None:
            traversible_[goal_map == 1] = 1
        else:
            traversible_[dilated_goal_map == 1] = 1
        vis_planner = FMMPlanner(traversible_)
        curr_loc_map = np.zeros_like(goal_map)
        # Update our location for finding the closest goal
        curr_loc_map[start[0], start[1]] = 1
        # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
        vis_planner.set_multi_goal(curr_loc_map)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        # find closest point on non-dilated goal map
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def get_closest_goal(self, goal_map, start):
        """closest goal, avoiding any obstacles."""
        empty = np.ones_like(goal_map)
        empty_planner = FMMPlanner(empty)
        empty_planner.set_goal(start)
        dist_map = empty_planner.fmm_dist * goal_map
        dist_map[dist_map == 0] = 10000
        closest_goal_map = dist_map == dist_map.min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        # You must move at least 5 cm when doing forward actions
        # Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.col_width += 2
            if self.col_width == 7:
                length = 4
                buf = 3
            self.col_width = min(self.col_width, 5)
        else:
            self.col_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        if dist < self.collision_threshold:
            # We have a collision
            width = self.col_width

            # Add obstacles to the collision map
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1

            return True
        
        return False