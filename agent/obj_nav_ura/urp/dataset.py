from torch.utils.data import Dataset
import os
import torch
from torch.utils.data import DataLoader

TEST_SCENE_IDS = []
MAX_DATA_PER_EP = 50
MIN_DATA_PER_EP = 1


class URPDataset(Dataset):

    def __init__(self, data_dir, split, data_config) -> None:
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

    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, index: int):
        data = torch.load(self.data_list[index])
        flat_idx, p, global_exp_pos_map_frame, c_s, i_s = data
        p = p.float() # convert from float16 to float32
        if self.data_config.type == "voxel_dense_map":
            map_size = self.data_config.map_size
            height = self.data_config.voxel_height
            voxel = torch.full([map_size, map_size,height], fill_value=-1, dtype=torch.float32) # x,y,z
            
            idx = torch.stack([flat_idx // (map_size*map_size), 
                               (flat_idx % (map_size*map_size)) // map_size, 
                               flat_idx % map_size], dim=-1).long()
            voxel[idx[:,0], idx[:,1], idx[:,2]] = p 
            info_map = torch.full([2, map_size, map_size], fill_value=0, dtype=torch.float32) # x,y
            info_map[0, global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = c_s
            info_map[1, global_exp_pos_map_frame[:,0], global_exp_pos_map_frame[:,1]] = i_s
            return voxel, info_map, global_exp_pos_map_frame
        
        else:
            raise ValueError("Invalid data type")
        return data


def get_dataloader(data_dir, split, batch_size, num_workers, data_config, shuffle=True):
    dataset = URPDataset(data_dir, split, data_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return dataloader