"""
Data struction and utility functions for point clouds.
"""

from typing import Optional, Union, List
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer.points.rasterize_points import rasterize_points
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
)
import trimesh.transformations as tra

import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

import torch
import home_robot.utils.depth as du
from home_robot.utils import rotation as ru

from pytorch3d.ops import sample_farthest_points

from .opt_dr import OptDR
DEBUG = False

class PointCloudManager:
    """
    Assumes same number of points for each batch.
    pointcloud and camera pose are in hab world frame (y-right, z-up, x-forward)
    """
    def __init__(self, config):
        self.config = config
        self._points = None
        self.downsample_pc_k = 0 # int(1e4)

        self.global_map_size = config.AGENT.SEMANTIC_MAP.map_size_cm
        self.map_center_coord = self.global_map_size / 200 #  in meter
        self.map_resolution = config.AGENT.SEMANTIC_MAP.map_resolution

        self.camera_tilt = None
        self.camera_height = None

    def add_points(self, points, agent_pose):
        """
        Add points to the point cloud. The points are assumed to be in the agent base frame, 
        and will be transformed to the world frame defined by habitat (y-right, z-up, x-forward)
        Args:
            points: tensor of B x N x 3, in agent base frame, unit is meter
            agent_pose: tuple (x,y,theta), in agent base frame, unit is meter and radian            
        """

        # # first convert to a frame that is aligned with the global frame
        # T = torch.tensor([
        #     [0, 1, 0],
        #     [-1, 0, 0],
        #     [0, 0, 1]
        # ], dtype=torch.float, device=xyz.device)
        # PL = torch.matmul(xyz, T.transpose(1, 0))

        # # then rotate the frame to align with the agent frame
        # R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=raw_obs.compass)
        # xyz_global = torch.matmul(PL, torch.from_numpy(R).float().transpose(1, 0).to(xyz.device))

        # # then translate the frame to align with the agent frame
        # xyz_global[..., 0] += cur_pose[0] * 100
        # xyz_global[..., 1] += cur_pose[1] * 100

        # use the transform_pose_t function to do the above
        xyz_global = du.transform_pose_t(
            points, agent_pose , points.device
        )

        if self.downsample_pc_k > 0:
            xyz_global, _ = sample_farthest_points(xyz_global, K=self.downsample_pc_k)

        if self._points is None:
            self._points = xyz_global
        else:
            self._points = torch.cat([self._points, xyz_global], dim=1)

        if DEBUG:
            show_points(self._points[0])

    
    def get_local_pointcloud(self, lmb):
        """
        Args:
            lmb: tensor of size [B, 4], (xmin, xmax, ymin, ymax), in world frame, in pixel,
            note lmb is from the semantic map, whose center is map_size/2, and the unit is pixel
            we need to convert it to meter by multiplying with the cell size, and then center it at 0
        """
        # TODO: this only works for single batch
        # since the number of points can be different for each batch after selection
        # we should to return a list of point clouds
        bs = lmb.shape[0]
        lmb = lmb * self.map_resolution / 100.0 # convert to meter
        lmb = lmb - self.map_center_coord  # center at 0
        idx = ( self._points[:, :, 0] > lmb[:, 0] ) & ( self._points[:, :, 0] < lmb[:, 1] ) & \
                ( self._points[:, :, 1] > lmb[:,2] ) & ( self._points[:, :, 1] < lmb[:,3] )
        
        return self._points[idx].view(bs, -1, 3)

    def add_groud_plane(self):
        """
        Add a ground plane to the point cloud
        """
        pass

    def visualize(self):
        """
        Visualize the point cloud
        """
        show_points(self._points)

    def rasterize(self, camera_pose, agent_pose, lmb = None):
        """
        Args:
            camera_pose: tensor of size [B, 4, 4], in "scene world frame"! (unknown to agent)
            We should only use this to derive the tilt and height of the camera
            agent_pose: tensor of size [B, 3], in agent base frame, unit is meter and radian
            lmb: 4, (xmin, xmax, ymin, ymax), in world frame, unit is meter
        """
        if lmb is not None:
            points = self.get_local_pointcloud(lmb)
        else:
            points = self._points

        if self.camera_tilt is None:

            if camera_pose is not None:
                # TODO: make consistent between sim and real
                # hab_angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="YZX")
                # angles = pt.matrix_to_euler_angles(camera_pose[:, :3, :3], convention="ZYX")
                angles = tra.euler_from_matrix(camera_pose[:3, :3], "rzyx")
                
                self.camera_tilt = - angles[1]

                # Get the camera height
                self.camera_height = camera_pose[2, 3]
            else:
                # otherwise use the default values
                self.camera_tilt = 0
                self.camera_height = self.config.ENVIRONMENT.camera_height

            # # get camera frame defined in base frame
            # R = ru.get_r_matrix([0.0, 1., 0.0], angle=-self.camera_tilt)
            # T_bc = torch.eye(4, 4)
            # T_bc[:3, :3] = torch.from_numpy(R).float()
            # T_bc[:3, 3] = torch.tensor([0, 0, self.camera_height])
            # self.T_bc = T_bc.to(points.device)
        
        # camera_pose = self.get_camera_frame(agent_pose)
        odr = OptDR(self.camera_tilt, self.camera_height, self.config, points.device)
        odr.init_agent_pose(torch.tensor(agent_pose).unsqueeze(0))
        odr(points)
        # rasterize_pc(points,
        #              agent_pose=agent_pose,
        #              camera_tilt=self.camera_tilt, 
        #              camera_height=self.camera_height,
        #              config=self.config)

    # def get_camera_frame(self, agent_pose):
    #     """
    #     get the camera frame expressed in the world frame
    #     TODO: support batch
    #     Args:
    #         agent_pose: tuple (x,y,theta), in agent base frame, unit is meter and radian
    #     """
    #     T_wb = get_agent_pose_matrix(agent_pose).to(self.T_bc.device) # base frame expressed in world frame
    #     T_wc = T_wb @ self.T_bc # camera frame expressed in world frame
    #     return T_wc

    



        