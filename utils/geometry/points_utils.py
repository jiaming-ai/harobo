

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

DEBUG = False


def show_points(points,feats=None):
    point_cloud = Pointclouds(points=[points], features=feats)
    fig = plot_scene({
        "Pointcloud": {
            "scene": point_cloud,
        }
    })
    fig.show()

def show_points_with_logit(points,feat):
    """
    points: N x 3
    feat: N x 1, logits
    """
    color = torch.zeros_like(points) # N x 3
    # > logit(0.5)
    color[feat.squeeze(1) >= 0] = torch.tensor([1,0,0],dtype=torch.float32, device=points.device) # red
    # < logit(0.5) and > logit(0.2)
    color[(feat.squeeze(1) <0) & (feat.squeeze(1)>=-1.39) ] = torch.tensor([0,0,1],dtype=torch.float32, device=points.device) # blue
    point_cloud = Pointclouds(points=[points], features=[color])
    fig = plot_scene({
        "Pointcloud": {
            "scene": point_cloud,
        }
    })
    fig.show()

def show_points_with_prob(points,feat):
    """
    points: N x 3
    feat: N x 1, logits
    """
    color = torch.zeros_like(points) # N x 3
    # > logit(0.5)
    color[feat.squeeze(1) >= 0.5] = torch.tensor([1,0,0],dtype=torch.float32, device=points.device) # red
    # < logit(0.5) and > logit(0.2)
    color[(feat.squeeze(1) <0.5) & (feat.squeeze(1)>=0.2) ] = torch.tensor([0,0,1],dtype=torch.float32, device=points.device) # blue
    point_cloud = Pointclouds(points=[points], features=[color])
    fig = plot_scene({
        "Pointcloud": {
            "scene": point_cloud,
        }
    })
    fig.show()

def show_voxel(voxel_tensor):
    """
    voxel_tensor: B x Z x X x Y
    """
    
    voxel = voxel_tensor[0].permute(2,1,0) # in hab world frame (x: forward, y: left, z: up)
    idx = torch.nonzero(~voxel.isinf()) # [N, 3]
    feat = voxel[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(1) # [N, 1]
    pc = idx.float() # [N, 3]
    # pc[:,:2] -= self.global_map_size / 2 
    # pc = pc * self.resolution + self.resolution / 2 # [N, 3] 
    # pc = pc / 100 # [N, 3] in meter    

    show_points_with_prob(pc,feat)

def get_pc_from_voxel(voxel, res, agent_pose=np.array([0.,0.,0.])):
    """
    get point cloud from voxel
    args:
        voxel: tensor of size [B, X, Y, Z], in agent base frame, the coordinates are in map frame
    """
    pc = torch.nonzero(~voxel.isinf()) # [N, 3]
    pc = pc.float() * res + res / 2 # [N, 3] in cm

    pc = du.transform_pose_t(
            pc, agent_pose , pc.device
        )
    
    return pc
    
def get_agent_pose_matrix(agent_pose):
    """
    get 4x4 transformation matrix describing agent pose. using habitat world frame (y-right, z-up, x-forward)

    Args:
        agent_pose: tuple (x,y,theta), in agent base frame, unit is meter and radian, theta is counter-clockwise
    """
    x, y, theta = agent_pose
    R = ru.get_r_matrix([0.0, 0.0, 1.0], angle=theta)
    T = torch.eye(4, 4)
    T[:3, 3] = torch.tensor([x, y, 0.0])
    T[:3, :3] = torch.tensor(R)
    return T
    
# def convert_camera_to_pytorch3d_frame(camera_pos_in_hab):
#     """
#     convert camera pose from habitat frame to pytorch3d WORLD frame (x-right, y-up, z-forward)
#     Args:
#         camera_pos_in_hab: B x 4 x 4, in habitat frame
#     """
#     T = torch.tensor([
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 0]
#     ], dtype=torch.float32, device=camera_pos_in_hab.device)
#     camera_pos_in_pt3d = camera_pos_in_hab.clone()
#     camera_pos_in_pt3d[:,:3,:] = T @ camera_pos_in_hab[:, :3,:] # B x 3 x 4
#     return camera_pos_in_pt3d

def convert_to_pytorch3d_frame(coords_in_hab):
    """
    convert coords from habitat world frame (y-right, x-forward, z-up),
    to pytorch3d WORLD frame (x-right, y-up, z-forward)
    Args:
        coords_in_hab: B x N x 3, in habitat frame
    """
    T = torch.tensor([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=torch.float32, device=coords_in_hab.device)
    coords_in_pt3d = coords_in_hab @ T.transpose(0, 1) # B x N x 3
    return coords_in_pt3d


def rasterize_pc(points,agent_pose,camera_tilt,camera_height, settings=None, config=None):
    """
    All input coords are in habitat frame, we need to convert them to pytorch3d frame for rendering
    Args:
        points: B x N x 3
        camera_pose: B x 4 x 4
    """

    # first we need to transform the points to the world frame used by torch3d
    points = convert_to_pytorch3d_frame(points) # pt3d world frame
    point_cloud = Pointclouds(points=points, features=None)
    
    if settings is None:
        raster_settings = PointsRasterizationSettings(
            image_size=(config.ENVIRONMENT.frame_height, config.ENVIRONMENT.frame_width), 
            radius = 0.02,
            points_per_pixel = 1
        )
    else:
        raster_settings = settings

    # R = torch.tensor([
    #     [0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 0]
    # ]).unsqueeze(0)    
    # T = torch.tensor([0,0,0]).unsqueeze(0)
    # camera_pos_pt3d_frame = convert_camera_to_pytorch3d_frame(camera_pose) # B x 4 x 4
    # R = camera_pos_pt3d_frame[:,:3,:3] # B x 3 x 3
    # T = -camera_pos_pt3d_frame[:,:3, 3] # B x 3 
    # R = torch.eye(3).unsqueeze(0)
    # R = torch.tensor(ru.get_r_matrix([1.0, 0.0, 0.0], angle=-tilt)).unsqueeze(0)

    # TODO: support batched camera pose
    tilt = torch.tensor(camera_tilt)
    theta = torch.tensor(agent_pose[2])
    Rx = torch.tensor([
        [1, 0, 0],
        [0, tilt.cos(), -tilt.sin()],
        [0, tilt.sin(), tilt.cos()]
    ], dtype=torch.float32, device=points.device).unsqueeze(0)
    Ry = torch.tensor([
        [theta.cos(), 0, theta.sin()],
        [0, 1, 0],
        [-theta.sin(), 0, theta.cos()]
    ], dtype=torch.float32, device=points.device).unsqueeze(0)

    R = Ry @ Rx

    # calculate T, which is the coords of the world origin in the camera frame
    # T = -R^T @ t_wc, where t_wc is the coords of the camera in the world frame
    t_wc = torch.tensor([0, camera_height, 0], dtype=torch.float32)
    t_wc[[2,0]] = torch.tensor(agent_pose[:2])
    t_wc = t_wc.unsqueeze(0).to(points.device)
    T = -R.transpose(1, 2) @ t_wc.unsqueeze(-1)
    T = T.squeeze(-1)

    

    # R, T = look_at_view_transform(eye=torch.tensor([[0,0.3,0]],dtype=torch.float32),
    #                               at=torch.tensor([[0,0.3,2]],dtype=torch.float32),
    #                             )
    T._requires_grad = True

    cameras = FoVPerspectiveCameras(znear=0.01,#config.ENVIRONMENT.min_depth,
                                    # zfar=config.ENVIRONMENT.max_depth, 
                                    fov=config.ENVIRONMENT.hfov, 
                                    # aspect_ratio=config.ENVIRONMENT.frame_height/config.ENVIRONMENT.frame_width,
                                    device=points.device,
                                    R=R,
                                    T=T,
                                    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)

    

    idx, zbuf, dists2 = rasterizer(point_cloud)

    plt.figure(figsize=(10, 10))
    plt.imshow(zbuf[0, ..., 0].cpu().numpy())
    plt.axis("off")

