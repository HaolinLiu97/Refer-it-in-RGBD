import torch.nn as nn
class res_mlp_block(nn.Module):
    def __init__(self,num_channel,use_2d=False):
        super().__init__()
        if use_2d==False:
            self.mlp1=nn.Conv1d(num_channel,num_channel,kernel_size=1)
            self.mlp2=nn.Conv1d(num_channel,num_channel,kernel_size=1)
        else:
            self.mlp1 = nn.Conv2d(num_channel, num_channel, kernel_size=1)
            self.mlp2 = nn.Conv2d(num_channel, num_channel, kernel_size=1)
    def forward(self,input):
        net=input
        net=self.mlp1(net)
        net=self.mlp2(net)
        net=F.relu(input+net)
        return net
class res_mlp(nn.Module):
    def __init__(self,num_channel,num_layer,use_2d=False):
        super().__init__()
        self.res_block_list=nn.ModuleList()
        for i in range(num_layer):
            self.res_block_list.append(res_mlp_block(num_channel,use_2d=use_2d))
    def forward(self,input):
        net=input
        for i in range(len(self.res_block_list)):
            net=self.res_block_list[i](net)
        return net