    
from habitat.utils.visualizations import fog_of_war, maps
import numpy as np
import torch
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

class IGRayCasting():

    def __init__(self,config,device) -> None:
        # self.config = config
        self.camera_hfov = config.ENVIRONMENT.hfov # in degrees
        self.prior = config.AGENT.SEMANTIC_MAP.probability_prior
        self.vr = config.ENVIRONMENT.max_depth // config.AGENT.SEMANTIC_MAP.map_resolution * 100 # in grids

        num_view = int(360 // self.camera_hfov + 1)
        turn_angle = 360 / num_view
        self.look_angle = [i * turn_angle for i in range(num_view)]
        self.device = device
    
    def cal_info_gains(self,global_obstacle_map,global_exp_map,global_prob_map,global_coords):
        """
        Compute the information gain for global_coords in global_map using 2D ray casting
        args:
            global_obstacle_map: binery obstacle map of numpy array, shape: (M,M)
            global_exp_map: binery exploration map of numpy array, shape: (M,M)
            global_prob_map: probability map of numpy array, shape: (M,M)
            global_coords: numpy array, shape: (N,2)
        """
        
        i_s = np.zeros(len(global_coords))
        c_s = np.zeros(len(global_coords))
        promising_grids = global_prob_map > self.prior
        global_map = (global_obstacle_map == 0)
        for i in range(len(global_coords)):
        
            mask = np.zeros_like(global_obstacle_map)

            for angle in self.look_angle:
                mask = fog_of_war.reveal_fog_of_war(
                    top_down_map=global_map,
                    current_fog_of_war_mask=mask,
                    current_point=global_coords[i],
                    current_angle=angle,
                    fov=self.camera_hfov,
                    max_line_len=self.vr,
                )
            unseen_area = np.logical_and(mask==1, global_exp_map==0)
            c_s[i] = np.sum(unseen_area) * self.prior  # coverage information gain
            i_s[i] = np.sum(np.logical_and(mask, promising_grids)*global_prob_map) # information information gain
        
        c_s = c_s / ( np.pi * self.vr**2 / 2 ) # normalize by the area of the circle
        i_s = i_s / ( np.pi * self.vr**2 / 2 * self.prior) # normalize by the area of the circle
        return torch.from_numpy(c_s).to(self.device), torch.from_numpy(i_s).to(self.device)