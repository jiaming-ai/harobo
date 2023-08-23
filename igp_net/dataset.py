from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader
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



TEST_SCENE_IDS = ['102344049','102816009']
MAX_DATA_PER_EP = 50
MIN_DATA_PER_EP = 1


class URPDataset(Dataset):

    def __init__(self, data_dir, split, data_config, device='cpu') -> None:
        super().__init__()
        train_data_list = []
        test_data_list = []
        for scene_id in os.listdir(data_dir):
            scene_dir = os.path.join(data_dir, scene_id)
            for ep_id in os.listdir(scene_dir):
                ep_dir = os.path.join(scene_dir, ep_id)
                data_files = [os.path.join(ep_dir,f) for f in os.listdir(ep_dir) if f.endswith(".pt")][:MAX_DATA_PER_EP]
                if len(data_files) < MIN_DATA_PER_EP:
                    continue
                if scene_id in TEST_SCENE_IDS:
                    test_data_list.extend(data_files)
                else:
                    train_data_list.extend(data_files)

        if split == "train":
            self.data_list = train_data_list
        elif split == "test":
            self.data_list = test_data_list
        else:
            raise ValueError("Invalid split")
        
        self.data_config = data_config
        print(f"Loaded {len(self.data_list)} data files for {split} split")

        if data_config.random_rotate:
            print("Warning: random rotate is enabled")

        self.device = torch.device(device)

    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int):
        data = torch.load(self.data_list[index])
        if len(data) == 5:
            flat_idx, p, global_exp_pos_map_frame, c_s, i_s = data
        elif len(data) == 6:
            flat_idx, p, global_exp_pos_map_frame, c_s, i_s, theta_list = data
            n_views = len(theta_list)
            c_s = c_s.view(-1, n_views).sum(dim=-1)
            i_s = i_s.view(-1, n_views).sum(dim=-1)
        else:
            raise ValueError("Invalid data format")
        
        flat_idx = flat_idx.to(self.device)
        p = p.to(self.device)
        global_exp_pos_map_frame = global_exp_pos_map_frame.to(self.device)
        c_s = c_s.to(self.device)
        i_s = i_s.to(self.device)

        p = p.float() # convert from float16 to float32
        if self.data_config.type == "voxel_dense_map":
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

            info_map = torch.full([2, map_size, map_size], fill_value=-1, dtype=torch.float32,device=self.device) # x,y
            info_map[0, global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = c_s
            info_map[1, global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = i_s * self.data_config.i_s_weight # scale i_s to make it more visible
            # print(f"c_s avg: {c_s.mean()}, i_s avg: {i_s.mean()}, x: {c_s.mean()/i_s.mean()}")


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
                info_map = torch.nn.MaxPool2d(xy_scale)(info_map)

            if z_scale > 1:
                voxel = torch.nn.MaxPool1d(z_scale)(voxel.permute(1,2,0)).permute(2,0,1)
                
            # random rotate
            if random_rotate:
                angle = float(torch.empty(1).uniform_(float(-180), float(180)).item())
                voxel = TF.rotate(voxel, angle,fill=-1)
                info_map = TF.rotate(info_map, angle,fill=-1)
                
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
                info_map = info_map[:, x_min:x_max, y_min:y_max]

                if voxel.shape[1] < crop_size or voxel.shape[2] < crop_size:
                    voxel = T.Pad((crop_size-voxel.shape[2],  crop_size-voxel.shape[1],0,0),fill=-1)(voxel)
                    info_map = T.Pad((crop_size-info_map.shape[2], crop_size-info_map.shape[1],0,0),fill=-1)(info_map)
            
            # voxel = voxel.permute(1,2,0) # z,x,y -> x,y,z
          
            # if voxel.max() > 0.5:
            #     print('show here')
            return voxel, info_map

        else:
            raise ValueError("Invalid data type")


def get_dataloader(data_dir, split, batch_size, num_workers, data_config, device='cpu',shuffle=True):
    dataset = URPDataset(data_dir, split, data_config,device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader