
import torch
from torch.nn import functional as F

# def get_affine_grid_batch(translation, theta, scale, out_size, device):
#     """
#     Get the affine transformation matrix defined by pos, theta and scale

#     ### Parameters:
#         - translation: torch.tensor of size: N * 2 for x, y
#         - theta: torch.tensor of size: N * 1 theta = arctan y/x
#         - scale: torch.tensor of size: N * 2 for  scale_x and scale y
#         - out_size: list of [N, C, W, H] 
#     ### Return:
#         - torch.tensor of size N * 2 * 3
#     """
#     bs = translation.shape[0]
#     zeros = torch.zeros([bs,1],device=device)
#     ones = torch.ones([bs,1],device=device)
#     scale_matrix_r1 = torch.concat([scale[:,:1],zeros,zeros],dim=1)
#     scale_matrix_r2 = torch.concat([zeros,scale[:,1:2],zeros],dim=1)
#     r3 = torch.concat([zeros,zeros,ones],dim=1)
#     scale_matrix = torch.stack([scale_matrix_r1,scale_matrix_r2,r3],dim=1)

#     cos_theta = theta.cos()
#     sin_theta = theta.sin()
#     rot_mt_r1 = torch.concat([cos_theta,-sin_theta,zeros],dim=1)
#     rot_mt_r2 = torch.concat([sin_theta,cos_theta,zeros],dim=1)
#     rot_matrix = torch.stack([rot_mt_r1,rot_mt_r2,r3],dim=1)

#     trans_mt_r1 = torch.concat([ones,zeros,translation[:,0:1]],dim=1)
#     trans_mt_r2 = torch.concat([zeros,ones,translation[:,1:2]],dim=1)
#     trans_matrix = torch.stack([trans_mt_r1,trans_mt_r2,r3], dim=1)

#     # first scale, then rotation, then translation
#     rot_scale = torch.bmm(scale_matrix,rot_matrix)
#     trans_rot_scale = torch.bmm(rot_scale, trans_matrix)

#     grid = F.affine_grid(trans_rot_scale[:,:2,:],out_size)
#     return grid

def get_affine_grid_batch(translation, theta, scale, out_size, device):
    """
    Get the affine transformation matrix defined by pos, theta and scale

    ### Parameters:
        - translation: torch.tensor of size: [N, 2] for x, y
        - theta: torch.tensor of size: [N, 1] theta = arctan y/x
        - scale: torch.tensor of size: [N, 2] for  scale_x and scale y
        - out_size: list of [N, C, W, H] 
    ### Return:
        - torch.tensor of size N * 2 * 3
    """

    bs = translation.shape[0]
    zeros = torch.zeros([bs,1],device=device)
    ones = torch.ones([bs,1],device=device)
    scale_matrix_r1 = torch.concat([scale[:,:1],zeros,zeros],dim=1)
    scale_matrix_r2 = torch.concat([zeros,scale[:,1:2],zeros],dim=1)
    r3 = torch.concat([zeros,zeros,ones],dim=1)
    scale_matrix = torch.stack([scale_matrix_r1,scale_matrix_r2,r3],dim=1)

    cos_theta = theta.cos()
    sin_theta = theta.sin()
    rot_mt_r1 = torch.concat([cos_theta,-sin_theta,zeros],dim=1)
    rot_mt_r2 = torch.concat([sin_theta,cos_theta,zeros],dim=1)
    rot_matrix = torch.stack([rot_mt_r1,rot_mt_r2,r3],dim=1)

    trans_mt_r1 = torch.concat([ones,zeros,translation[:,0:1]],dim=1)
    trans_mt_r2 = torch.concat([zeros,ones,translation[:,1:2]],dim=1)
    trans_matrix = torch.stack([trans_mt_r1,trans_mt_r2,r3], dim=1)

    # first scale, then rotation, then translation???
    # trans_rot_scale = scale_matrix @ rot_matrix @ trans_matrix
    trans_rot_scale = trans_matrix @ (scale_matrix @ rot_matrix) # WHY TSR but not TRS?? 
    # (1. guess: the parenthesis is not necessary)
    # 2. the scale is defined in rotated frame?

    affine_trans_mat = trans_rot_scale[:,:2,:]
    grid = F.affine_grid(affine_trans_mat,out_size,align_corners=False)
    return grid

def get_affine_grid(translation, theta, scale, out_size, device):
    """
    Get the affine transformation matrix defined by pos, theta and scale

    ### Parameters:
        - translation: torch.tensor of size: [2] for x, y
        - theta: torch.tensor of size: [1] theta = arctan y/x
        - scale: torch.tensor of size: [2] for  scale_x and scale y
        - out_size: list of [1, C, W, H] 
    ### Return:
        - torch.tensor of size N * 2 * 3
    """
    assert out_size[0] == 1, 'Use get_affine_grid_batch for batched inputs'

    zeros = torch.zeros([1],device=device)
    ones = torch.ones([1],device=device)
    scale_matrix_r1 = torch.concat([scale[:1],zeros,zeros])
    scale_matrix_r2 = torch.concat([zeros,scale[1:2],zeros])
    r3 = torch.concat([zeros,zeros,ones])
    scale_matrix = torch.stack([scale_matrix_r1,scale_matrix_r2,r3],dim=0)

    cos_theta = theta.cos()
    sin_theta = theta.sin()
    rot_mt_r1 = torch.concat([cos_theta,-sin_theta,zeros],dim=0)
    rot_mt_r2 = torch.concat([sin_theta,cos_theta,zeros],dim=0)
    rot_matrix = torch.stack([rot_mt_r1,rot_mt_r2,r3],dim=0)

    trans_mt_r1 = torch.concat([ones,zeros,translation[0:1]],dim=0)
    trans_mt_r2 = torch.concat([zeros,ones,translation[1:2]],dim=0)
    trans_matrix = torch.stack([trans_mt_r1,trans_mt_r2,r3], dim=0)

    # first scale, then rotation, then translation???
    # trans_rot_scale = scale_matrix @ rot_matrix @ trans_matrix
    trans_rot_scale = trans_matrix @ (scale_matrix @ rot_matrix) # WHY TSR but not TRS??

    affine_trans_mat = trans_rot_scale[:2,:].unsqueeze(0)
    grid = F.affine_grid(affine_trans_mat,out_size,align_corners=False)
    return grid

# def get_affine_grid_tr(translation, theta, out_size, device):
#     """
#     Get the affine transformation matrix defined by pos, theta.
#     tr means translate first then rotation (no scaling)

#     ### Parameters:
#         - translation: torch.tensor of size: [2] for x, y
#         - theta: torch.tensor of size: [1] theta = arctan y/x
#         - out_size: list of [1, C, W, H] 
#     ### Return:
#         - torch.tensor of size N * 2 * 3
#     """
#     assert out_size[0] == 1, 'Use get_affine_grid_batch for batched inputs'

#     zeros = torch.zeros([1],device=device)
#     ones = torch.ones([1],device=device)
#     r3 = torch.concat([zeros,zeros,ones])

#     cos_theta = theta.cos()
#     sin_theta = theta.sin()
#     rot_mt_r1 = torch.concat([cos_theta,-sin_theta,zeros],dim=0)
#     rot_mt_r2 = torch.concat([sin_theta,cos_theta,zeros],dim=0)
#     rot_matrix = torch.stack([rot_mt_r1,rot_mt_r2,r3],dim=0)

#     trans_mt_r1 = torch.concat([ones,zeros,translation[0:1]],dim=0)
#     trans_mt_r2 = torch.concat([zeros,ones,translation[1:2]],dim=0)
#     trans_matrix = torch.stack([trans_mt_r1,trans_mt_r2,r3], dim=0)

#     # trans_rot_scale = rot_matrix @trans_matrix 
#     trans_rot_scale = trans_matrix 

#     affine_trans_mat = trans_rot_scale[:2,:].unsqueeze(0)
#     grid = F.affine_grid(affine_trans_mat,out_size)
#     return grid


def get_affine_grid_no_scaling(translation, theta, out_size, device):
    """
    Get the affine transformation matrix defined by pos, theta and scale

    ### Parameters:
        - translation: torch.tensor of size: N * 2 for x, y
        - theta: torch.tensor of size: N * 1 theta = arctan y/x
        - out_size: list of [N, C, W, H] 
    ### Return:
        - torch.tensor of size N * 2 * 3
    """
    

    bs = translation.shape[0]
    zeros = torch.zeros([bs,1],device=device)
    ones = torch.ones([bs,1],device=device)
    r3 = torch.concat([zeros,zeros,ones])

    cos_theta = theta.cos()
    sin_theta = theta.sin()
    rot_mt_r1 = torch.concat([cos_theta,-sin_theta,zeros])
    rot_mt_r2 = torch.concat([sin_theta,cos_theta,zeros])
    rot_matrix = torch.stack([rot_mt_r1,rot_mt_r2,r3],dim=1)

    trans_mt_r1 = torch.concat([ones,zeros,translation[:,0]])
    trans_mt_r2 = torch.concat([zeros,ones,translation[:,1]])
    trans_matrix = torch.stack([trans_mt_r1,trans_mt_r2,r3], dim=1)

    trans_rot = torch.bmm(rot_matrix, trans_matrix)

    grid = F.affine_grid(trans_rot[:,2,:],out_size)
    return grid