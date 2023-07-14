

from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer.points.rasterize_points import rasterize_points
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    # PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras,
)

import trimesh.transformations as tra
from .render import PulsarPointsRenderer
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

import torch
import home_robot.utils.depth as du
from home_robot.utils import rotation as ru

from pytorch3d.ops import sample_farthest_points
import torch.nn as nn
from .points_utils import convert_to_pytorch3d_frame
from .points_utils import show_points

class OptDR(nn.Module):

    def __init__(self, camera_tilt, camera_height, config, device) -> None:
        super().__init__()
        self.device = device
        self.camera_tilt = torch.tensor(camera_tilt)
        self.camera_height = torch.tensor(camera_height)
        self.camera_hfov = config.ENVIRONMENT.hfov
        self.screen_width = config.ENVIRONMENT.frame_width
        self.screen_height = config.ENVIRONMENT.frame_height

        self.config = config
        self.Rx = torch.tensor([
            [1, 0, 0],
            [0, self.camera_tilt.cos(), -self.camera_tilt.sin()],
            [0, self.camera_tilt.sin(), self.camera_tilt.cos()]
        ], dtype=torch.float32, device=self.device).unsqueeze(0) # [1, 3, 3]

        
        self.raster_settings = PointsRasterizationSettings(
            image_size=(self.screen_height, self.screen_width), 
            radius = 0.02,
            points_per_pixel = 10
        )
        self.visualize = False
        self.agent_pose = None
        self.cameras = FoVPerspectiveCameras(znear=0.01,#config.ENVIRONMENT.min_depth,
                                    # zfar=config.ENVIRONMENT.max_depth, # TODO: how to correctly set this? need to be tight
                                    fov=self.camera_hfov, 
                                    # aspect_ratio=self.screen_width/self.screen_height, # 4/3,TODO do we need this?
                                    device=self.device,
                                    # R=self.R,
                                    # T=self.T,
                                    )
        self.pulsar_cameras = FoVPerspectiveCameras(znear=0.01,#config.ENVIRONMENT.min_depth,
                                    zfar=config.ENVIRONMENT.max_depth, # TODO: how to correctly set this? need to be tight
                                    fov=self.camera_hfov, 
                                    aspect_ratio=self.screen_width/self.screen_height, # 4/3,TODO do we need this?
                                    device=self.device,
                                    # R=self.R,
                                    # T=self.T,
                                    )
        self.rasterizer = PointsRasterizer(raster_settings=self.raster_settings,
                                           cameras=self.cameras)


    def init_agent_pose(self, agent_pose):
        """
        args:
            agent_pose: tensor of size [N, 3] of [x, y, theta]
        """
        if type(agent_pose) == np.ndarray:
            agent_pose = torch.from_numpy(agent_pose).float()
        assert agent_pose.shape[1] == 3, "agent_pose should be of size [N, 3] of [x, y, theta]"

        self.agent_pose = nn.Parameter(agent_pose, requires_grad=True).to(self.device)
        
        
        
    def forward(self, points):
        """
        args:
            points: tensor of size [N, 3] of [x, y, z] in the world frame defined in the hab frame
        """
        if self.agent_pose is None:
            raise ValueError("agent_pose not initialized. Please call init_agent_pose() first")
        
        points = convert_to_pytorch3d_frame(points) # pt3d world frame
        point_cloud = Pointclouds(points=points, features=None)
        R, T = self.get_camera_matrix(self.agent_pose) # [N, 3, 3], [N, 3]


        idx, zbuf, dists2 = self.rasterizer(point_cloud, R=R, T=T)

       

        # test with depth
        # first transform the points to the camera frame
        points_in_cam_frame = points @ R.transpose(1,2) + T.unsqueeze(-1).transpose(1,2)
        dist = torch.norm(points_in_cam_frame, dim=-1,keepdim=True) # [N, P, 1]
        renderer = PulsarPointsRenderer(
            rasterizer=PointsRasterizer(cameras=self.pulsar_cameras, raster_settings=self.raster_settings),
            n_channels=1,
        ).to(self.device)
        pc_dist = Pointclouds(points=points, features=dist)
        images, rets = renderer(pc_dist, 
                                gamma=(1e-4,),
                                # max_depth = 10,
                                return_forward_info =True,
                                R=R.transpose(1,2),
                                T=T,)
        # zbuf_pulsar = renderer.renderer.depth_map_from_result_info_nograd(rets[0])
        if self.visualize:
            plt.figure(figsize=(10, 10))
            plt.imshow(images[0,...,0].cpu().numpy())
            plt.imshow(zbuf[0, ..., 0].cpu().numpy())
            # plt.axis("off")
            
        # return idx, zbuf, dists2
    
    def get_camera_matrix(self, agent_pose : torch.tensor) -> torch.Tensor:
        """
        get the camera matrix from the agent pose.
        Note agent pose is the pose of the agent in the world frame defined in hab frame.
        by the Hartley & Zisserman convention, the camera matrix is the matrix that transforms
        a point in the world frame to the camera frame. i.e. R = R_c^-1, T = -R_c^-1 * T_c, where
        R_c is the rotation matrix of the camera frame in the world frame, and T_c is the coords of
        the world origin in the camera frame.
        It seems pytorch3d is not following this convention: pt_in_cam_frame = pt_in_world_frame @ R + T
        Note here pt is [B, N, 3], so this right mul is actually by the transpose of R.
        args:
            agent_pose: tensor of size [N, 3] of [x, y, theta]
        """
        # first calculate R, which is the rotation matrix from the world frame to the camera frame
        theta = agent_pose[:, 2] # [N]
        cos_theta = theta.cos() # [N]
        sin_theta = theta.sin() # [N]
        zeros = torch.zeros_like(theta) # [N]
        ones = torch.ones_like(theta) # [N]

        Ry = torch.stack([cos_theta, zeros, sin_theta,
                          zeros, ones, zeros,
                          -sin_theta, zeros, cos_theta], dim=1).reshape(-1, 3, 3) # [N, 3, 3] 
    

        R = Ry.to(self.device) @ self.Rx # [N, 3, 3]

        # calculate T, which is the coords of the world origin in the camera frame
        # T = -R^T @ t_wc, where t_wc is the coords of the camera in the world frame
        t_wc = torch.full_like(agent_pose, fill_value=self.camera_height) # [N, 3]
        t_wc[:,[2,0]] = agent_pose[:, :2]
        t_wc = t_wc.to(self.device)
        T = -R.transpose(1,2) @ t_wc.unsqueeze(-1)
        T = T.squeeze(-1)
        
        return R, T