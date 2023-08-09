

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
from .points_utils import show_points,show_points_with_prob, show_points_with_logit
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

# from .pulsar_renderer import PulsarPointsRenderer
from .pulsar_unified import PulsarPointsRenderer

class DRPlanner(nn.Module):
    """
    Uncertainty reduction planning by differentiable rendering
    We use Pulsar renderer for rendering (https://arxiv.org/abs/2004.07484)
    """

    def __init__(self, camera_tilt, camera_height, config, device) -> None:
        super().__init__()
        self.device = device
        self.camera_tilt = torch.tensor(camera_tilt)
        # TODO: how to correctly set this? need to be tight
        self.camera_height = torch.tensor(config.ENVIRONMENT.camera_height + 0.16)
        self.camera_hfov = config.ENVIRONMENT.hfov
        self.screen_width = config.ENVIRONMENT.frame_width
        self.screen_height = config.ENVIRONMENT.frame_height
        self.prior = config.AGENT.SEMANTIC_MAP.probability_prior
        self.vr = config.ENVIRONMENT.max_depth
        
        self.config = config
        self.Rx = torch.tensor([
            [1, 0, 0],
            [0, self.camera_tilt.cos(), -self.camera_tilt.sin()],
            [0, self.camera_tilt.sin(), self.camera_tilt.cos()]
        ], dtype=torch.float32, device=self.device).unsqueeze(0) # [1, 3, 3]

        
        self.raster_settings = PointsRasterizationSettings(
            image_size=(self.screen_height, self.screen_width), 
            radius = 0.07,
            points_per_pixel = 5
        )
        self.visualize = False
        # self.agent_pose = None

        self.pulsar_cameras = FoVPerspectiveCameras(
                                    fov=self.camera_hfov, 
                                    aspect_ratio=self.screen_width/self.screen_height, # 4/3,TODO do we need this?
                                    device=self.device,                                
                                    )
       
        self.renderer = PulsarPointsRenderer(
            rasterizer=PointsRasterizer(cameras=self.pulsar_cameras, raster_settings=self.raster_settings),
            n_channels=1,
            max_num_spheres=1e6,
        ).to(self.device)

        
        # # testing
        self.cameras = FoVPerspectiveCameras(
                                    fov=self.camera_hfov, 
                                    device=self.device,
                                    )
        self.rasterizer = PointsRasterizer(raster_settings=self.raster_settings,
                                           cameras=self.cameras)
        self.renderer_pt3d  = PointsRenderer(
                            rasterizer=self.rasterizer,
                            compositor=AlphaCompositor(background_color=(-1,))
                            )
        
    # def init_agent_pose(self, agent_pose):
    #     """
    #     args:
    #         agent_pose: tensor of size [N, 3] of [x, y, theta]
    #     """
    #     if type(agent_pose) == np.ndarray:
    #         agent_pose = torch.from_numpy(agent_pose).float()
    #     assert agent_pose.shape[1] == 3, "agent_pose should be of size [N, 3] of [x, y, theta]"

    #     self.agent_pose = nn.Parameter(agent_pose.to(self.device), requires_grad=True)
        
        
    def _process_rendered_info(self, images):
        """
        args:
            images: tensor of size [B, H, W]
        """
        no_hit = (images==-1) # bg is -1
        
        too_far = torch.full_like(images, False, dtype=torch.bool)
        too_far[:,:50,...] = True # TODO: how to correctly set this? need to be tight. currently this is for vr=10m
        
        # case 1: no hit and too far (> vr=10m)
        # images[no_hit & too_far] = 0 # no info

        # case 2: no hit but not too far, then we assume we can obtain some info
        coverage_score = torch.sum(no_hit & (~too_far), dim=[1,2]) * self.prior / (self.screen_height*self.screen_width) 

        # case 3: hit, and prob < prior, then we assume no info
        # images[images<self.prior] = 0 # no info
        
        # case 4: hit, and prob > prior, then we assume we can obtain some info
        of_interest = (images > self.prior).float()
        interest_score = (images * of_interest).sum(dim=[1,2]) / (self.screen_height*self.screen_width)

        # print("coverage_score", coverage_score)
        # print("interest_score", interest_score)

        return coverage_score, interest_score


    def render(self, points, feat, agent_pose):
        """
        args:
            points: tensor of size [B, N, 3] of [x, y, z] in the world frame defined in the hab frame
        """
        
        points = convert_to_pytorch3d_frame(points) # pt3d world frame
        feat = torch.sigmoid(feat) # [B, N, 1]
        R, T = self.get_camera_matrix(agent_pose) # [N, 3, 3], [N, 3]
        
        # test batch rendering
        import time
        start = time.time()
        bs = 1
        R = R.repeat(bs,1,1)
        T = T.repeat(bs,1) * 10 
        points = points.expand(bs,-1,-1) * 10 # multiply by 10 to make the znear effectively become 0.1m, while keeping f as 1
        gamma = [1e-4 for _ in range(R.shape[0])] # we can use small gamma if we don't need differentiable depth
        znear = [1.0 for _ in range(R.shape[0])] # this is effectively 0.1m
        zfar = [100.0 for _ in range(R.shape[0])] # this is effectively 10m
        fov = [self.camera_hfov for _ in range(R.shape[0])]
        aspect_ratio = [self.screen_width/self.screen_height for _ in range(R.shape[0])]

        pc = Pointclouds(points=points, features=feat)
        images, rets = self.renderer.forward(pc,
                                                gamma=gamma,
                                                return_forward_info =True,
                                                znear=znear,
                                                zfar=zfar,
                                                fov=fov,
                                                aspect_ratio=aspect_ratio,
                                                R=R,
                                                max_n_hits=5,
                                                T=T,)
        print("pulsar rendering time: ", time.time() - start)

        # post process
        # # first set pixels <0 to 0
        # no_hit = images<0
        # too_far = torch.full_like(images, False, dtype=torch.bool)
        # too_far[:,:50,...] = True

        # images[no_hit & too_far] = 0
        # images[no_hit & (~too_far)] = self.prior

        self._process_rendered_info(images.squeeze(-1))

        # images, rets = self.renderer.batch_render(points,
        #                                           dist,
        #                                         gamma=1e-3,
        #                                         return_forward_info =True,
        #                                         R=R,
        #                                         T=T,)
        depth = self.renderer.renderer.depth_map_from_result_info_nograd(rets[0])

        start = time.time()
        images_pt = self.renderer_pt3d(pc, R=R, T=T)
        print("pytorch3d rendering time: ", time.time() - start)

        fragment = self.rasterizer(pc, R=R, T=T)
        zbuf = fragment.zbuf

        if self.visualize:
            plt.figure(figsize=(10, 10),)
            plt.imshow(images[0,...,0].detach().cpu().numpy())
            plt.figure(figsize=(10, 10))
            plt.imshow(np.flipud(depth.detach().cpu().numpy()))

            plt.figure(figsize=(10, 10))
            plt.imshow(images_pt[0, ..., 0].detach().cpu().numpy())

            plt.figure(figsize=(10, 10))
            plt.imshow(zbuf[0, ..., 0].detach().cpu().numpy())
            
        # return idx, zbuf, dists2
    
    def _filter_points(self,points, feats, agent_pose):
        """
        args:
            points: tensor of size [1, P, 3] of [x, y, z] in the world frame defined in the hab frame
            feats: tensor of size [1, P, 1]
            agent_pose: tensor of size [N, 2] of [x, y]
            
        """
        n_views = agent_pose.shape[0]
        points = points.repeat(n_views, 1, 1) # [N, P, 3]
        ap = agent_pose.unsqueeze(1).repeat(1, points.shape[1], 1) # [N, P, 2]
        within_vr = (points[:, :, 0] > (ap[:, :, 0] - self.vr)) & \
                    (points[:, :, 0] < (ap[:, :, 0] + self.vr )) & \
                    (points[:, :, 1] > (ap[:, :, 1] - self.vr )) & \
                    (points[:, :, 1] < (ap[:, :, 1] + self.vr )) 
        
        return within_vr
        
        # points = points[within_vr]
        # feats = feats.repeat(n_views, 1, 1)[within_vr]
        # return points, feats
        
        
    def cal_info_gains(self, points, feats, agent_pose):
        """
        args:
            points: tensor of size [1, P, 3] of [x, y, z] in the world frame defined in the hab frame
            feats: tensor of size [1, P, 1]
            agent_pose: tensor of size [N, 2] of [x, y]
        """
        # feats = torch.sigmoid(feats) # [1, P, 1], converts to probs
        # points = convert_to_pytorch3d_frame(points) * 10 # pt3d world frame, [1, P, 3]

        # R, T, n_view_per_loc = self.get_panoramic_camera_matrix(agent_pose) # [bs, 3, 3], [bs, 3]
        # N = agent_pose.shape[0] # this is the number of locations
        # bs = R.shape[0] # this is the total number of views needed to be rendered

        # # filter out points that are too far
        # vr_idx = self._filter_points(points, feats, agent_pose) # [N, P, 3], [N, P, 1]
        # point_list = []
        # feat_list = []
        # for i in range(vr_idx.shape[0]):
        #     point_list.extend(points[0][vr_idx[i]] * n_view_per_loc)
        #     feat_list.extend(feats[0][vr_idx[i]] * n_view_per_loc)

        points = convert_to_pytorch3d_frame(points) *10 # pt3d world frame
        feats = torch.sigmoid(feats) # [1, P, 1], converts to probs

        R, T, n_view_per_loc = self.get_panoramic_camera_matrix(agent_pose) # [bs, 3, 3], [bs, 3]
        N = agent_pose.shape[0] # this is the number of locations
        bs = R.shape[0] # this is the total number of views needed to be rendered

        points = points.expand(bs,-1,-1) 
        feats = feats.expand(bs,-1,-1)

        T = T * 10
        gamma = [1e-4 for _ in range(bs)] # we can use small gamma if we don't need differentiable depth
        znear = [1 for _ in range(bs)]
        zfar = [100.0 for _ in range(bs)]
        fov = [self.camera_hfov for _ in range(bs)]
        aspect_ratio = [self.screen_width/self.screen_height for _ in range(bs)]

        pc = Pointclouds(points=points, features=feats)
        # pc = Pointclouds(points=point_list, features=feat_list)
        
        images = self.renderer.forward(pc,
                                            gamma=gamma,
                                            znear=znear,
                                            zfar=zfar,
                                            fov=fov,
                                            aspect_ratio=aspect_ratio,
                                            R=R,
                                            max_n_hits=5,
                                            T=T,)
        c_s, i_s = self._process_rendered_info(images.squeeze(-1))
        info_at_locs = (c_s+10*i_s).view(N, n_view_per_loc).sum(-1) # [N_locs]

        


        if self.visualize:
            images, rets = self.renderer.forward(pc,
                                            gamma=gamma,
                                            return_forward_info =True,
                                            znear=znear,
                                            zfar=zfar,
                                            fov=fov,
                                            aspect_ratio=aspect_ratio,
                                            R=R,
                                            max_n_hits=5,
                                            T=T,)
            n, m = 3, 5
            _, axes = plt.subplots(n, m, figsize=(12, 18))
            for i in range(n):
                for j in range(m):
                    depth = self.renderer.renderer.depth_map_from_result_info_nograd(rets[i*m+j])
                    axes[i,j].imshow(np.flipud(depth.detach().cpu().numpy()))

            _, axes = plt.subplots(n, m, figsize=(12, 18))
            for i in range(n):
                for j in range(m):
                    axes[i,j].imshow(images[i*m+j,...,0].detach().cpu().numpy())


        return info_at_locs

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

        Note:
            R is the rotation matrix of the camera frame in the world frame
            T is the coords of the world origin in the camera frame
        args:
            agent_pose: tensor of size [N, 3] of [x, y, theta], units are meters and radians
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
    

    def get_panoramic_camera_matrix(self, agent_pose: torch.tensor) -> torch.tensor:
        """
        Generate camera matrics for panoramic rendering.
        Note: the camera frame is defined in the hab frame.

        Args:
            agent_pose: tensor of size [N, 3] of [x, y, theta] or [N, 2] of [x, y]
        """
        # TODO: how to solve the overlap problem?
        n_view_per_loc = int(360 / self.camera_hfov) + 1
        bs = agent_pose.shape[0]

        # TODO: vectorize this
        # tehta_tensor = torch.linspace(0, 2*np.pi, n_view_per_loc).unsqueeze(0).repeat(bs,1).to(agent_pose.device) # [N, n_view_per_loc]
        # agent_pose_tensor = agent_pose.unsqueeze(1).repeat(1, n_view_per_loc, 1) # [N, n_view_per_loc, 3]
        # agent_pose_tensor[:,:,2] = tehta_tensor # [N, n_view_per_loc, 3]
        # R_vec, T_vec = self.get_camera_matrix(agent_pose_tensor.view(-1, 3)) # [N*n_view_per_loc, 3, 3], [N*n_view_per_loc, 3]
     

        R_list = []
        T_list = []
        fov_rad = self.camera_hfov * np.pi / 180
        for i in range(n_view_per_loc):
            theta = torch.tensor([i * fov_rad]).unsqueeze(0).repeat(bs,1).to(agent_pose.device) # [1,1]
            agent_pose = torch.cat([agent_pose[:,:2], theta], dim=1) # [N, 3]
            R, T = self.get_camera_matrix(agent_pose)
            R_list.append(R)
            T_list.append(T)
        
        R = torch.stack(R_list, dim=1).view(-1, 3, 3) # [bs*n_view_per_loc, 3, 3]
        T = torch.stack(T_list, dim=1).view(-1, 3)

        return R, T, n_view_per_loc