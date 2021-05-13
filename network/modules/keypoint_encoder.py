import torch.nn
import torch
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
import torch.nn as nn

class keypoint_encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 output_channels=256,
                 npoint=[2048,1024,512],
                 radius=[0.2,0.4,0.8],
                 nsample=[64,32,16],
                 mlp=[[3,64,64,128],[128,128,128,256],[256,256,256,256]],
                 bn=True,
                 ret_first=False):
        super().__init__()
        self.ret_first=ret_first
        mlp[0][0]=in_channels
        mlp[-1][-1]=output_channels
        self.in_channels=int(in_channels)
        self.SA_Module=nn.ModuleList()
        self.npoint=npoint
        self.radius=radius
        self.nsample=nsample
        self.mlp=mlp
        for i in range(len(self.npoint)):
            self.SA_Module.append(PointnetSAModuleVotes(
                npoint=self.npoint[i],
                radius=self.radius[i],
                nsample=self.nsample[i],
                mlp=self.mlp[i],
                use_xyz=True,
                normalize_xyz=True,
                bn=bn
            ))

    def forward(self,input):
        batch_size=input.shape[0]

        xyz=input[:,0:3,:].transpose(1,2).contiguous() # B,N,3
        if self.in_channels>0:
            features=input[:,3:,:].contiguous() # B,C,N
        else:
            features=None
        coor_dict={}

        for i in range(len(self.SA_Module)):
            xyz,features,ind=self.SA_Module[i](xyz,features)
            if i==0:
                old_ind=ind.long()
                coor_dict["xyz_2048"]=xyz
                coor_dict["ind_2048"]=old_ind
            else:
                old_ind=old_ind[torch.arange(batch_size)[:,None],ind.long()]
                if i==1:
                    coor_dict["xyz_1024"] = xyz
                    coor_dict["ind_1024"] = old_ind
                if i==2:
                    coor_dict["xyz_512"]=xyz
                    coor_dict["ind_512"]=old_ind
                if i==3:
                    coor_dict["xyz_256"]=xyz
                    coor_dict["ind_256"]=old_ind

        return xyz,features,coor_dict