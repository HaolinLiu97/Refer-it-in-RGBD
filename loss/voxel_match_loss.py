import torch
import torch.nn as nn

class voxel_match_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion=nn.MSELoss()
    def forward(self,output,label):
        positive_ind=torch.where(label>0.2)
        positive_loss=self.criterion(output[positive_ind[0],positive_ind[1]],label[positive_ind[0],positive_ind[1]])
        negative_ind=torch.where(label<=0.2)
        negative_loss=self.criterion(output[negative_ind[0],negative_ind[1]],label[negative_ind[0],negative_ind[1]])
        loss=positive_loss+negative_loss
        loss=loss/2
        return loss