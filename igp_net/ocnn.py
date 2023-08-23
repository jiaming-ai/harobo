

import torch
import torch.nn as nn
import ocnn.octree as octree
import ocnn
from ocnn.octree import Octree



class OCLeNet(torch.nn.Module):
    r''' Octree-based LeNet for classification.
    '''

    def __init__(self, in_channels: int, stages: int,
                nempty: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.stages = stages
        self.nempty = nempty
        channels = [in_channels] + [2 ** max(i+7-stages, 2) for i in range(stages)]

        self.convs = torch.nn.ModuleList([ocnn.modules.OctreeConvBnRelu(
            channels[i], channels[i+1], nempty=nempty) for i in range(stages)])
        self.pools = torch.nn.ModuleList([ocnn.nn.OctreeMaxPool(
            nempty) for i in range(stages)])
        self.octree2voxel = ocnn.nn.Octree2Voxel(self.nempty)
        # self.header = torch.nn.Sequential(
        #     torch.nn.Dropout(p=0.5),                     # drop1
        #     ocnn.modules.FcBnRelu(64 * 64, 128),         # fc1
        #     torch.nn.Dropout(p=0.5),                     # drop2
        #     torch.nn.Linear(128, out_channels))          # fc2

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        r''''''

        for i in range(self.stages):
            d = depth - i
            data = self.convs[i](data, octree, d)
            data = self.pools[i](data, octree, d)
        data = self.octree2voxel(data, octree, depth-self.stages)
        # data = self.header(data)
        return data


class OCResNet(nn.Module):
    def __init__(self, in_channels: int, resblock_num: int,
               stages: int, nempty: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.resblk_num = resblock_num
        self.stages = stages
        self.nempty = nempty
        channels = [2 ** max(i+9-stages, 2) for i in range(stages)]

        self.conv1 = ocnn.modules.OctreeConvBnRelu(
            in_channels, channels[0], nempty=nempty)
        self.pool1 = ocnn.nn.OctreeMaxPool(nempty)
        self.resblocks = torch.nn.ModuleList([ocnn.modules.OctreeResBlocks(
            channels[i], channels[i+1], resblock_num, nempty=nempty)
            for i in range(stages-1)])
        self.pools = torch.nn.ModuleList([ocnn.nn.OctreeMaxPool(
            nempty) for i in range(stages-1)])
        self.global_pool = ocnn.nn.OctreeGlobalPool(nempty)

        # self.header = torch.nn.Linear(channels[-1], out_channels, bias=True)
        # self.header = torch.nn.Sequential(
        #     ocnn.modules.FcBnRelu(channels[-1], 512),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(512, out_channels))

    def forward(self, data: torch.Tensor, octree: Octree, depth: int):
        r''''''

        data = self.conv1(data, octree, depth)
        data = self.pool1(data, octree, depth)
        for i in range(self.stages-1):
            d = depth - i - 1
            data = self.resblocks[i](data, octree, d)
            data = self.pools[i](data, octree, d)
        data = self.global_pool(data, octree, depth-self.stages)
        # data = self.header(data)
        return data
