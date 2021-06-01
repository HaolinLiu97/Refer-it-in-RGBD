import torch
import torch.nn as nn
FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3

l2_criterion=nn.MSELoss()
def huber_loss(error, delta=1.0):
    """
    Args:
        error: Torch tensor (d1,d2,...,dk)
    Returns:
        loss: Torch tensor (d1,d2,...,dk)

    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    0.5 * d^2 + d * (|x|-d)     if |x|>d
    Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/models/model_util.py
    """
    abs_error = torch.abs(error)
    #quadratic = torch.min(abs_error, torch.FloatTensor([delta]))
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    loss = 0.5 * quadratic**2 + delta * linear
    return loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False,keepdim=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]

    #pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, 1, 1) #B,N,M,C
    #pc2_expand_tile = pc2.unsqueeze(1).repeat(1, 8, 1, 1) #B,N,M,C
    #print(pc1_expand_tile.shape,pc2_expand_tile.shape)

    pc1_expand_tile = pc1.unsqueeze(2).repeat(1, 1, M, 1) #B,N,M,C
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1, N, 1, 1) #B,N,M,C
    pc_diff = pc1_expand_tile - pc2_expand_tile #B,N,M,C

    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1)  # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1)  # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff ** 2, dim=-1)  # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2,keepdim=keepdim)  # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1,keepdim=keepdim)  # (B,M)
    return dist1, idx1, dist2, idx2