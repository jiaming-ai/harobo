#!/usr/bin/env python3

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from habitat_sim.utils.viz_utils import semantic_to_rgb
import imageio
from habitat.utils.visualizations import maps
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from habitat.utils.visualizations.utils import images_to_video
import os

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene

import numpy as np
import torch
from pytorch3d.ops import sample_farthest_points


def render_plt_image(data):
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvasAgg(fig)
    
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    ax.imshow(data,cmap='coolwarm',vmin=0,vmax=1.2)
    # ax.axis('off')
    # plt.gcf().delaxes(plt.gca())
    # fig.tight_layout()
    canvas.draw()  # Draw the canvas, cache the renderer

    buf = canvas.buffer_rgba()
    # convert to a NumPy array
    image = np.asarray(buf)
    return image

def save_image(sm,file_name=None,dir=None,add_time=False):
    if dir is None:
        dir = 'out/'
    if file_name is None:
        file_name = 'tmp.png'
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
    parts = file_name.split('.')
    name = parts[0]
    if len(parts)==2:
        ext = parts[1]
    else:
        ext = 'png'
    if add_time:
        t = datetime.now().strftime('%d_%H_%M')
        file_name = f'{name}_{t}.{ext}'
    else:
        file_name = f'{name}.{ext}'

    # plt.imshow(sm)
    # plt.savefig(f'{dir}/{file_name}')
    imageio.imsave(f'{dir}/{file_name}',sm)

def save_img_tensor(tensor,fn,dir=None):
    if tensor.shape[0]==3:
        img = tensor.permute(1,2,0)
    else:
        img = tensor
    save_image(img.detach().cpu().numpy(),fn,dir)
    
def plot_image(sm):
    """
    plot image using matplot lib
    """
    # sm = semantic_to_rgb(sm)
    
    imgplot = plt.imshow(sm)
    
    plt.show()

def plot_multiple_images(images,row=1):

    """
    plot multiple images using matplot lib
    """
    col = len(images) // row
    _, axes = plt.subplots(row, col, figsize=(12, 18))
    for i in range(row):
        for j in range(col):
            np_img = images[i*col+j]
            if isinstance(np_img, torch.Tensor):
                np_img = np_img.cpu().detach().numpy()
            axes[i,j].imshow(np_img)
    plt.show()

def display_grayscale(image):
    img_bgr = np.repeat(image, 3, 2)
    cv2.imshow("Depth Sensor", img_bgr)
    return cv2.waitKey(0)


def display_rgb(image):
    img_bgr = image[..., ::-1]
    cv2.imshow("RGB", img_bgr)
    return cv2.waitKey(0)

def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def img_frombytes(data):
    """
    Used to solve a PIL bug that can't import bool array properly
    """
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def annotate_map_with_goal(map, goal):
    """
    Annotate the map with the goal location
    """
    if goal is not None:
        draw = ImageDraw.Draw(map)
        y, x = goal
        r = 5
        if map.mode =='L':
            color = 118
        else:
            color = (0, 255, 0,255)
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    return map

def tile_images(images):
    r"""Tile multiple images into single image

    Args:
        images: list of images where each image has dimension
            (height x width x channels)

    Returns:
        tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    np_images = np.asarray(images)
    n_images, height, width, n_channels = np_images.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    np_images = np.array(
        images
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = np_images.reshape(
        new_height, new_width, height, width, n_channels
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image

def visualize_pred(rgb, 
                   depth, 
                   gt_local, 
                   pred_local, 
                   vis_obj_dict,
                   bsm=None,
                   gt_bsm=None,
                   annotate_plan=None,
                   **kwargs):
    """
    Visualize the predicted local map
    Parameters:
        rgb: tensor of size (C, H, W) or np [h, w, 3]
        depth: tensor of size (1, H, W), or np [h, w, 1]
        gt_local: tensor of size (C, vr, vr)
        pred_local: tensor of size (C, vr, vr)
        vis_obj_dict: dict of {idx: obj_class}
        bsm: tensor of [ h, w]
        gt_bsm: tensor of [ h, w]
        high_goal: ndarray of [h, w]
    """
    if annotate_plan is not None and 'sem_vis' in annotate_plan:
        rgb = annotate_plan['sem_vis']
        
    if type(rgb) == torch.Tensor:
        imw, imh = rgb.shape[1], rgb.shape[2]

        rgb = (rgb.cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
        depth = (depth.cpu().squeeze(0).numpy()*255).astype(np.uint8)
    else:
        # directly from env observation
        imw, imh = rgb.shape[0], rgb.shape[1]
        depth = (depth.squeeze(2)*255).astype(np.uint8)
        rgb = rgb.astype(np.uint8)

    vr = pred_local.shape[1]
    height = int(imw * 2 + 60) # margin is 20

    pred_row_num = int((height - 20) / (vr+20))
    pred_col_num = int(len(vis_obj_dict) // pred_row_num + 1 )

    col_width = vr*2 + 30
    width = int(imw + 40 + pred_col_num * (col_width))

    if bsm is not None:
        bsm_w = int (imh / bsm.shape[0] * bsm.shape[1])
        width = int( width + bsm_w + 20 )
    img = Image.new("RGB",(width,height))

    img.paste(Image.fromarray(rgb),(20,20))
    img.paste(Image.fromarray(depth,mode='L'),(20,imh+40))

    col_num = 0
    row_num = 0
    w_start = imw+40
    h_start = (height - pred_row_num * (vr+20)) // 2 + 10
    row_height = vr+20
    lines = []
    obj_items = list(vis_obj_dict.items())
    for j,_ in obj_items:
        offset = 2 if len(gt_local) < len(pred_local) else 0
        gt_img = (gt_local[j].cpu().numpy()*255).astype(np.uint8)
        pred_img = (pred_local[j+offset].detach().cpu().numpy()*255).astype(np.uint8)

        pred_img = np.rot90(pred_img,2)
        gt_img = np.rot90(gt_img,2)
        img.paste(Image.fromarray(gt_img,mode='L'),(int(w_start+col_num*col_width),int(h_start+row_num*row_height)))
        img.paste(Image.fromarray(pred_img,mode='L'),(int(w_start+col_num*col_width+vr+10),int(h_start+row_num*row_height)))
        line = [
            (int(w_start+col_num*col_width),int(h_start+row_num*row_height+vr+15)),
            (int(w_start+col_num*col_width+vr*2+10),int(h_start+row_num*row_height+vr+15))
        ]
        lines.append(line)

        row_num += 1
        if row_num >= int(pred_row_num-1):
            row_num = 0
            col_num += 1
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("utils/fonts/CaviarDreams.ttf", 13)
    for i, l in enumerate(lines):
        draw.line(l,fill='white', width=0)
        draw.text((l[0][0]+20,l[0][1]-14),obj_items[i][1],font=font)

    if bsm is not None:
        bsm = (bsm.detach().cpu().numpy()*255).astype(np.uint8)
        gt_bsm = (gt_bsm.detach().cpu().permute(1,2,0).numpy()).astype(np.uint8)
        if annotate_plan:
            # zoom in
            x1,x2,y1,y2 = annotate_plan['lmb']
            bsm = bsm[x1:x2,y1:y2]
            gt_bsm = gt_bsm[x1:x2,y1:y2]
            
        new_size = (imh,bsm_w)
        bsm_img = Image.fromarray(bsm,mode='L')
        gt_img = Image.fromarray(gt_bsm,mode='RGB')
        if annotate_plan:
            high_goal = annotate_plan['closest_goal_pt']
            bsm_img = annotate_map_with_goal(bsm_img,high_goal)
            gt_img = annotate_map_with_goal(gt_img,high_goal)

        bsm_img = bsm_img.resize(new_size)
        gt_img = gt_img.resize(new_size)
        img.paste(bsm_img,(width-20-bsm_w,20))
        img.paste(gt_img,(width-20-bsm_w,imh+40))

    img = np.array(img)
    return img


def visualize_gt(rgb,depth,map, gt_local,pred_local):
    """
    Visualize gt maps for testing
    height: 480*2 + 20*2 (20 margin) + 10
    width: 20 + 640 + 10 + 
    """
    # 
    height = 1010
    map_ratio = map.shape[0] / map.shape[1] # height / width
    
    map_height = 800
    map_width = int(map_height / map_ratio)
    
    if map_width > 600:
        map_width = 600
        map_height = int(map_width * map_ratio)

    map_y = 20

    gt_local_width = gt_local.shape[1]
    gt_local_margin = (map_width - gt_local_width*2) // 3
    if gt_local_margin < 0:
        gt_local_margin = 0
        map_x = 670 + (gt_local_width*2-map_width)//2
        total_width = 690 + gt_local_width*2
    else:
        map_x = 670
        total_width = map_x + map_width + 20

    total_width = map_x + map_width + 20 
    img = Image.new("RGBA",(total_width,height)) # height: 480*2 + 20*2 + 10
    img.paste(Image.fromarray(rgb),(20,20))
    img.paste(Image.fromarray((depth*255).squeeze(2).astype(np.uint8),mode='L'),(20,510))
    
    resize_map = Image.fromarray(map.astype(np.uint8))
    
    resize_map = resize_map.resize((map_width, map_height),Image.BICUBIC)
    img.paste(resize_map,(map_x, map_y)) # 20 + 640 + 10 = 670, 20
    
    gt_local_y = height - 20 - gt_local.shape[0]
    img.paste(img_frombytes(gt_local.astype(bool)), (670+gt_local_margin, gt_local_y)) # 670 + 10 + width + 10
    img.paste(img_frombytes(pred_local.astype(bool)), (670+gt_local_margin*2+gt_local_width, gt_local_y)) # 670 + 10 + width + 10 + gt_local_width + 10

    img = np.array(img)

    return img

class Recording():
    def __init__(self) -> None:
        self._images = []

    # def add_frame(self,rgb,depth,map, gt_local,pred_local):
    #     img = visualize_gt(rgb,depth,map, gt_local,pred_local)
    #     self._images.append(img)
    #     return img

    def add_frame(self,img):
        self._images.append(img)

    def save_video(self,fname, dir='recordings'):
        images_to_video(self._images, dir, fname)
        self._images = []



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
    color[(feat.squeeze(1) <0.5) & (feat.squeeze(1)>=1e-3) ] = torch.tensor([0,0,1],dtype=torch.float32, device=points.device) # blue
    point_cloud = Pointclouds(points=[points], features=[color])
    fig = plot_scene({
        "Pointcloud": {
            "scene": point_cloud,
        }
    })
    fig.show()

def show_voxel_with_logit(voxel_tensor):
    """
    voxel_tensor: B x Z x X x Y
    """
    if voxel_tensor.dim() == 4:
        voxel_tensor = voxel_tensor[0]
    
    voxel = voxel_tensor.permute(2,1,0) # in hab world frame (x: forward, y: left, z: up)
    idx = torch.nonzero(~voxel.isinf()) # [N, 3]
    feat = voxel[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(1) # [N, 1]
    feat = torch.sigmoid(feat)
    pc = idx.float() # [N, 3]
    # pc[:,:2] -= self.global_map_size / 2 
    # pc = pc * self.resolution + self.resolution / 2 # [N, 3] 
    # pc = pc / 100 # [N, 3] in meter    

    show_points_with_prob(pc,feat)

def show_voxel_with_prob(voxel_tensor):
    """
    voxel_tensor: B x Z x X x Y
    """
    if voxel_tensor.dim() == 4:
        voxel_tensor = voxel_tensor[0]
    
    voxel = voxel_tensor.permute(2,1,0) # in hab world frame (x: forward, y: left, z: up)
    idx = torch.nonzero(~torch.logical_or(voxel.isinf(),voxel==-1)) # [N, 3]
    feat = voxel[idx[:,0], idx[:,1], idx[:,2]].unsqueeze(1) # [N, 1]
    pc = idx.float() # [N, 3]
    # pc[:,:2] -= self.global_map_size / 2 
    # pc = pc * self.resolution + self.resolution / 2 # [N, 3] 
    # pc = pc / 100 # [N, 3] in meter    

    show_points_with_prob(pc,feat)