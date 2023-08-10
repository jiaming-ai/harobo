
import torch
import torch.nn as nn
import ocnn.octree as octree
import ocnn
from ocnn.octree import Octree
from .igp_net import OCResNet, OCLeNet


class IGPNet(nn.Module):
    """Information Gain Prediction Network
    """

    def __init__(self,config,net_config) -> None:
        super().__init__()
        self.config = config
        self.map_size_half_m = config.AGENT.SEMANTIC_MAP.map_size_cm / 100 / 2
        map_size_cell = config.AGENT.SEMANTIC_MAP.map_size_cm // config.AGENT.SEMANTIC_MAP.map_resolution
        self.net_config = net_config

        self.backbone_type = net_config['backbone']
        if self.backbone_type == 'ocresnet':
            self.backbone = OCResNet(1,3,4,True)
        elif self.backbone_type == 'oclenet':
            self.backbone = OCLeNet(1,4,True)
        elif self.backbone_type == 'pointnet':
            pass
        elif self.backbone_type == 'conv2d':
            pass
        else:
            raise NotImplementedError

    def get_octree_from_points(self, points):
        """
        args:
            points: tensor [N,3], (x,y,z) in hab world frame, in meters
        """
        # scale points to [-1,1]
        points = points / self.map_size_half_m
        depth = self.net_config['depth']
        ot = octree.Octree(depth)
        ot.build_octree(points)
        return ot
    
    def get_voxel_from_points(self, points):
        """
        args:
            points: tensor [N,3], (x,y,z) in hab world frame, in meters
        """
        voxel = torch.full((),fill_value=-1,dtype=torch.float32)
        
    def forward(self,points, pose):
        """
        args:
            points: tensor [N,3], (x,y,z) in hab world frame, in meters
            pos: tensor [N,3], (x,y,theta) in hab world frame, in meters, theta in degrees
        """

        # normalize input to [-1,1]
        pos = pose[:,:2] / self.map_size_half_m
        theta = ( pose[:,2] % 360 - 180 ) / 180
        x = torch.cat([pos,theta],dim=1)

        ot = self.get_octree_from_points(points)