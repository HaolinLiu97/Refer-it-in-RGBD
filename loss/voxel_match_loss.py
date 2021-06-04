import torch
import torch.nn as nn

class voxel_match_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion=nn.MSELoss()
    def forward(self,output,label):
        positive_mask=torch.zeros(label.shape).cuda()
        positive_mask=torch.where(label>0.2,torch.ones_like(positive_mask), positive_mask)
        positive_loss=self.criterion(output*positive_mask,label*positive_mask)
        negative_mask=torch.zeros(label.shape).cuda()
        negative_mask = torch.where(label <= 0.2, torch.ones_like(negative_mask), negative_mask)
        negative_loss=self.criterion(output*negative_mask,label*negative_mask)
        loss=positive_loss+negative_loss
        loss=loss/2
        return loss