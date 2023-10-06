from .igp_net import IGPNet
import os
from omegaconf import OmegaConf
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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


class IGPredictor():

    def __init__(self,model_dir,device) -> None:
        config_path = f"{model_dir}/config.yaml"
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
        else:
            print('Warning: No config file found, using default config')
        
        self.device = device
                                        
        net = IGPNet(config.dataloader,config.net)
        net.load_state_dict(torch.load(f"{model_dir}/best.pth"))
        net.to(device)
        net.eval()

        self.net = net
        self.data_config = config.dataloader
        self.i_s_weight = config.dataloader.i_s_weight

    def _process_voxel(self,voxel):

        assert voxel.shape[0] == self.data_config.voxel_height, "voxel height is not correct"

        empty = voxel.isinf()
        voxel = torch.sigmoid(voxel)
        voxel[empty] = -1

        filter_height = self.data_config.filter_height
        min_height = self.data_config.filter_height_min
        max_height = self.data_config.filter_height_max
        xy_scale = self.data_config.xy_scale
        z_scale = self.data_config.z_scale
        map_size = self.data_config.map_size
        crop_size = self.data_config.crop_size

        if filter_height:
            if voxel[ :min_height, ...].max() >= 0:
                print("Warning: voxel min height is not enough")
            if voxel[ max_height:, ...].max() >= 0:
                print("Warning: voxel max height is not enough")
            voxel = voxel[ min_height:max_height, ...]
        # downsample
        if xy_scale > 1:
            voxel = torch.nn.MaxPool2d(xy_scale)(voxel)

        if z_scale > 1:
            voxel = torch.nn.MaxPool1d(z_scale)(voxel.permute(1,2,0)).permute(2,0,1)
            
            # crop
        if crop_size > 0:
            xy_proj = voxel.max(dim=0)[0] >= 0 # x,y
            xy_idx = torch.nonzero(xy_proj, as_tuple=True)
            x_min, x_max = xy_idx[0].min(), xy_idx[0].max()
            y_min, y_max = xy_idx[1].min(), xy_idx[1].max()

            x_length = x_max - x_min + 1
            y_length = y_max - y_min + 1
            if x_length > crop_size or y_length > crop_size:
                raise ValueError("crop size is too small")
            
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_min = max(0, x_center - crop_size // 2)
            x_max = min(map_size//xy_scale, x_center + crop_size // 2)
            y_min = max(0, y_center - crop_size // 2)
            y_max = min(map_size//xy_scale, y_center + crop_size // 2)
            voxel = voxel[:, x_min:x_max, y_min:y_max]

            if voxel.shape[1] < crop_size or voxel.shape[2] < crop_size:
                voxel = T.Pad((0,0,crop_size-voxel.shape[2],  crop_size-voxel.shape[1]),fill=-1)(voxel)
        
        return voxel.unsqueeze(0), xy_proj, x_min, y_min, x_max-x_min, y_max-y_min
    
    def process_pred(self,pred,xy_proj, x_min, y_min, x_len, y_len):
        
        
        xy_scale = self.data_config.xy_scale
        map_size = self.data_config.map_size
        crop_size = self.data_config.crop_size
        
        before_crop_size = map_size // xy_scale
        before_crop = torch.zeros(1,2,before_crop_size,before_crop_size, device=self.device)
        before_crop[...,x_min:x_min+x_len,y_min:y_min+y_len] = pred[..., :x_len, :y_len]

        # TODO: dialate?
        before_crop[~xy_proj.repeat(1,2,1,1)] = 0

        pred = torch.nn.Upsample(scale_factor=xy_scale, mode='nearest')(before_crop)

        return pred.squeeze(0)

    def _process_pc(self,flat_idx,p):
        
        
        flat_idx = flat_idx.to(self.device)
        p = p.float().to(self.device)

        filter_height = self.data_config.filter_height
        min_height = self.data_config.filter_height_min
        max_height = self.data_config.filter_height_max
        xy_scale = self.data_config.xy_scale
        z_scale = self.data_config.z_scale
        map_size = self.data_config.map_size
        height = self.data_config.voxel_height
        random_rotate = self.data_config.random_rotate
        crop_size = self.data_config.crop_size
        voxel = torch.full([height, map_size, map_size], fill_value=-1, dtype=torch.float32, device=self.device) # z,x,y
        
        idx = torch.stack([flat_idx // (map_size*map_size), 
                            (flat_idx % (map_size*map_size)) // map_size, 
                            flat_idx % map_size], dim=-1).long()
        voxel[idx[:,2], idx[:,1], idx[:,0]] = p 

        # filter height
        if filter_height:
            if voxel[ :min_height, ...].max() >= 0:
                print("Warning: voxel min height is not enough")
            if voxel[ max_height:, ...].max() >= 0:
                print("Warning: voxel max height is not enough")
            voxel = voxel[ min_height:max_height, ...]

        # downsample
        if xy_scale > 1:
            voxel = torch.nn.MaxPool2d(xy_scale)(voxel)

        if z_scale > 1:
            voxel = torch.nn.MaxPool1d(z_scale)(voxel.permute(1,2,0)).permute(2,0,1)
            
        # random rotate
        # no random rotate for point cloud
        # if random_rotate:
        #     angle = float(torch.empty(1).uniform_(float(-180), float(180)).item())
        #     voxel = TF.rotate(voxel, angle,fill=-1)
            
            # crop
        if crop_size > 0:
            xy_proj = voxel.max(dim=0)[0] >= 0 # x,y
            xy_idx = torch.nonzero(xy_proj, as_tuple=True)
            x_min, x_max = xy_idx[0].min(), xy_idx[0].max()
            y_min, y_max = xy_idx[1].min(), xy_idx[1].max()

            x_length = x_max - x_min + 1
            y_length = y_max - y_min + 1
            
            # if x_length > crop_size or y_length > crop_size:
            #     raise ValueError("crop size is too small")
            
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2
            x_min = max(0, x_center - crop_size // 2)
            x_max = min(map_size//xy_scale, x_center + crop_size // 2)
            y_min = max(0, y_center - crop_size // 2)
            y_max = min(map_size//xy_scale, y_center + crop_size // 2)
            voxel = voxel[:, x_min:x_max, y_min:y_max]

            if voxel.shape[1] < crop_size or voxel.shape[2] < crop_size:
                voxel = T.Pad((crop_size-voxel.shape[2],  crop_size-voxel.shape[1],0,0),fill=-1)(voxel)

        return voxel.unsqueeze(0), xy_proj, x_min, y_min, x_length, y_length

    @torch.no_grad()
    def predict_pc(self,flat_idx,p):
        """
        args:
            voxel: torch.tensor, shape: (H, M, M)
        """
        voxel,xy_proj, x_min, y_min, x_len, y_len = self._process_pc(flat_idx,p)
        voxel = voxel.to(self.device)
        pred = self.net(voxel)

        pred = self.process_pred(pred,xy_proj, x_min, y_min, x_len, y_len)
        return pred

    @torch.no_grad()
    def predict(self,voxel):
        """
        args:
            voxel: torch.tensor, shape: (H, M, M)
        """
        voxel,xy_proj, x_min, y_min, x_len, y_len = self._process_voxel(voxel)
        voxel = voxel.to(self.device)
        pred = self.net(voxel)

        pred = self.process_pred(pred,xy_proj, x_min, y_min, x_len, y_len)
        return pred