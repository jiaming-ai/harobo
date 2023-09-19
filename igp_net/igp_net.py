
import torch
import torch.nn as nn
# from .igp_net import OCResNet, OCLeNet
from .unet import UNetBackBone
from .resnet import ResNetBackbone
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


class IGPNet(nn.Module):
    """Information Gain Prediction Network
    """

    def __init__(self, data_config,net_config) -> None:
        super().__init__()
        self.map_size = data_config.crop_size
        self.height = (data_config.filter_height_max - data_config.filter_height_min )//2

        # fake_data = torch.zeros([1, self.height, self.map_size, self.map_size], dtype=torch.float32) # 
        self.i_s_weight = net_config.get('i_s_weight',0)
        
        self.backbone_type = net_config['backbone']
        if self.backbone_type == 'unet':
            self.backbone = UNetBackBone(self.height, 2, net_config.c0)
        elif self.backbone_type == 'resnet':
            self.backbone = ResNetBackbone(self.height, 2, data_config.crop_size, net_config.c0, net_config.resnet_depth)
        else:
            raise NotImplementedError
        
        if net_config.loss == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif net_config.loss == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError
        
    def forward(self, voxel):
        """
        args:
            voxel: tensor of shape [N,H,M,M], H is height, M is map size
        """
        return self.backbone(voxel)

    def loss(self, pred, info_map, reduction='mean'):
        has_val = info_map >= 0
        loss = self.loss_fn(pred, info_map)

        
        if self.i_s_weight > 0:
            i_map = info_map[:,1,...] > 1 # B x M x M
            emphasis = torch.zeros_like(loss) # B x 2 x M x M
            emphasis[:,1,...] = i_map # B x M x M
            loss[emphasis.bool()] *= self.i_s_weight # B x 2 x M x M
            
        if reduction == 'mean':
            loss = loss[has_val].mean()
        else:
            loss = loss[has_val]
            
        return loss

# class IGPNet(nn.Module):
#     """Information Gain Prediction Network
#     """

#     def __init__(self,config,net_config) -> None:
#         super().__init__()
#         self.config = config
#         self.map_size_half_m = config.AGENT.SEMANTIC_MAP.map_size_cm / 100 / 2
#         map_size_cell = config.AGENT.SEMANTIC_MAP.map_size_cm // config.AGENT.SEMANTIC_MAP.map_resolution
#         self.net_config = net_config

#         self.backbone_type = net_config['backbone']
#         if self.backbone_type == 'ocresnet':
#             self.backbone = OCResNet(1,3,4,True)
#         elif self.backbone_type == 'oclenet':
#             self.backbone = OCLeNet(1,4,True)
#         elif self.backbone_type == 'pointnet':
#             pass
#         elif self.backbone_type == 'conv2d':
#             pass
#         else:
#             raise NotImplementedError

#     def get_octree_from_points(self, points):
#         """
#         args:
#             points: tensor [N,3], (x,y,z) in hab world frame, in meters
#         """
#         # scale points to [-1,1]
#         points = points / self.map_size_half_m
#         depth = self.net_config['depth']
#         ot = octree.Octree(depth)
#         ot.build_octree(points)
#         return ot
    
#     def get_voxel_from_points(self, points):
#         """
#         args:
#             points: tensor [N,3], (x,y,z) in hab world frame, in meters
#         """
#         voxel = torch.full((),fill_value=-1,dtype=torch.float32)
        
#     def forward(self,points, pose):
#         """
#         args:
#             points: tensor [N,3], (x,y,z) in hab world frame, in meters
#             pos: tensor [N,3], (x,y,theta) in hab world frame, in meters, theta in degrees
#         """

#         # normalize input to [-1,1]
#         pos = pose[:,:2] / self.map_size_half_m
#         theta = ( pose[:,2] % 360 - 180 ) / 180
#         x = torch.cat([pos,theta],dim=1)

#         ot = self.get_octree_from_points(points)