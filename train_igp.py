from igp_net.dataset import get_dataloader
import yaml
from igp_net.igp_net import IGPNet
import argparse
from omegaconf import OmegaConf
import torch
from collections import deque
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import random

from utils.visualization import (
    display_grayscale,
    display_rgb,
    plot_image,
    plot_multiple_images,
    save_image, 
    draw_top_down_map, 
    Recording, 
    visualize_gt,
    visualize_pred,
    show_voxel_with_prob,
    save_img_tensor)

import os
DEBUG = os.environ.get("DEBUG", 0)
DEBUG = int(DEBUG)
# DEBUG = False

def vis(voxel,pred,gt):
    voxel = voxel[0].cpu().detach()
    pred = pred[0].cpu().detach()
    obstacle_height = 2
    exp_map = voxel[:obstacle_height,...].max(dim=0)[0] > 0
    occ_map = voxel[obstacle_height:,...].max(dim=0)[0] > 0
    c_map = pred[0]
    i_map = pred[1]
    c_map[~exp_map] = 0
    c_map[occ_map] = 0
    i_map[~exp_map] = 0
    i_map[occ_map] = 0

    images = [c_map,i_map,gt[0,0],gt[0,1]]
    plot_multiple_images(images,2)
    voxel[voxel==-1] = torch.inf
    show_voxel_with_prob(voxel)


def train_epoch(net,dataloader,optimizer,epoch,logger):
    net.train()
    losses = deque(maxlen=10)
    log_interval = 10

    for step, batch in enumerate(dataloader):
        voxel, info_map = batch
        # vis(voxel,info_map,info_map)
        optimizer.zero_grad()
        pred = net(voxel)
        loss = net.loss(pred,info_map)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % log_interval == 0:
            print(f"Epoch {epoch}, Step {step}, Loss {np.mean(losses)}")
            logger.add_scalar("train/loss",np.mean(losses),step+epoch*len(dataloader))

def eval_epoch(net,dataloader):
    net.eval()
    log_interval = 10
    all_loss = []
    for step, batch in enumerate(dataloader):
        voxel, info_map = batch
        pred = net(voxel)
        if DEBUG:
            if (info_map[0,1] > 1).sum() > 0:
                vis(voxel,pred,info_map)
        loss = net.loss(pred,info_map)
        all_loss.append(loss.item())

        if step % log_interval == 0:
            print(f"Eval Step {step}, Loss {np.mean(all_loss)}")
    
    avg_loss = np.mean(all_loss)
    print(f"Eval Loss {avg_loss}")
    return avg_loss


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/igp/default_config.yaml")
    parser.add_argument("--gpu_id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="unet_c16_lossis1_dlis10_more")
    parser.add_argument("--eval_only", action="store_true",default=False)
    parser.add_argument("--options", type=str,default='') # "net.c0=8,..."
    
    args = parser.parse_args()
    extra_args = OmegaConf.from_dotlist(args.options.split(","))
    
    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config,extra_args)
    
    if DEBUG:
        args.eval_only = True
        config.train.batch_size = 1

    
    init_seed(args.seed)
    if args.eval_only:
        eval(args,config)
    else:
        train(args,config)

def eval(args, config):

    
    model_dir = f"data/checkpoints/igp/{args.exp_name}"
    config_path = f"{model_dir}/config.yaml"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        print('Warning: No config file found, using default config')
    if DEBUG:
        config.train.batch_size = 1
        
    device = f"cuda:{args.gpu_id}"
    dataloader_eval = get_dataloader(data_dir="data/info_gain",
                                split="test",
                                batch_size=config.train.batch_size,
                                num_workers=0,
                                data_config=config.dataloader,
                                device = device,
                                shuffle=True)
                                     
    net = IGPNet(config.dataloader,config.net)
    net.load_state_dict(torch.load(f"{model_dir}/best.pth"))
    net.to(device)
    with torch.no_grad():
        loss = eval_epoch(net,dataloader_eval)
    print(f"Eval Loss {loss}")


def train(args,config):
    print(f'Training with config {config}')

    device = f"cuda:{args.gpu_id}"

    dataloader = get_dataloader(data_dir="data/info_gain",
                                split="train",
                                batch_size=config.train.batch_size,
                                num_workers=0,
                                data_config=config.dataloader,
                                device = device,
                                shuffle=True)
    dataloader_eval = get_dataloader(data_dir="data/info_gain",
                                split="test",
                                batch_size=config.train.batch_size,
                                num_workers=0,
                                data_config=config.dataloader,
                                device = device,
                                shuffle=True)
                                     
    net = IGPNet(config.dataloader,config.net)
    net.to(device)

    train_epoch_num = config.train.epoch_num
    optimizer = torch.optim.Adam(net.parameters(),lr=config.train.lr)

    logger = SummaryWriter(log_dir=f"datadump/logs/{args.exp_name}")
    best_loss = 100
    model_dir = f"data/checkpoints/igp/{args.exp_name}"
    os.makedirs(model_dir,exist_ok=True)
    OmegaConf.save(config,f"{model_dir}/config.yaml")
    
    for epoch in range(train_epoch_num):
        train_epoch(net,dataloader,optimizer, epoch, logger)
        with torch.no_grad():
            loss = eval_epoch(net,dataloader_eval)
        logger.add_scalar("eval/loss",loss,epoch)
        if loss < best_loss:
            best_loss = loss
            torch.save(net.state_dict(),f"{model_dir}/best.pth")
            print("Saved best model")




if __name__ == "__main__":
    main()

