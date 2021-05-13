import torch.nn as nn
import torch
import MinkowskiEngine as ME
from torch.nn.utils.rnn import pack_padded_sequence

class upblock(nn.Module):
    def __init__(self,down_channels,skip_channels,out_channels,D=3):
        super().__init__()
        self.deconv = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(down_channels, down_channels, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(skip_channels + down_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiReLU(inplace=True),
        )
    def forward(self,skip_feat,down_feat):
        up_feat=self.deconv(down_feat)
        cat_feat = ME.cat(up_feat, skip_feat)
        net = self.conv(cat_feat)

        return net

class UNet_fuselang(nn.Module):
    def __init__(self,en_ch=[3,64,128,256],de_ch=[256,128,32,1],D=3,lang_feat_dim=256):
        super().__init__()
        self.en1 = nn.Sequential(
            ME.MinkowskiConvolution(en_ch[0],en_ch[1],kernel_size=3,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2,stride=2,dimension=D)
        )  # 15
        self.en2=nn.Sequential(
            ME.MinkowskiConvolution(en_ch[1], en_ch[2], kernel_size=3,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2,stride=2,dimension=D)
        )
        self.en3=nn.Sequential(
            ME.MinkowskiConvolution(en_ch[2], en_ch[3], kernel_size=3,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiMaxPooling(kernel_size=2,stride=2,dimension=D)
        )
        self.fuse_layer = nn.Sequential(
            ME.MinkowskiConvolution(en_ch[3] + lang_feat_dim,en_ch[3] + lang_feat_dim,kernel_size=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(en_ch[3] + lang_feat_dim, en_ch[3] + lang_feat_dim // 2, kernel_size=1,dimension=D),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(en_ch[3] + lang_feat_dim // 2, en_ch[3], kernel_size=1,dimension=D),
            ME.MinkowskiReLU(inplace=True)
        )
        self.de3 = upblock(down_channels=en_ch[3], skip_channels=en_ch[2], out_channels=de_ch[0])
        self.de2 = upblock(down_channels=de_ch[0], skip_channels=en_ch[1], out_channels=de_ch[1])
        self.de1 = upblock(down_channels=de_ch[1], skip_channels=en_ch[0], out_channels=de_ch[2])
        self.final_layer = nn.Sequential(
            ME.MinkowskiConvolution(de_ch[2], de_ch[3], kernel_size=1,dimension=D),
        )

    def forward(self,input,lang_feat):
        x1=self.en1(input)
        x2=self.en2(x1)
        x3=self.en3(x2)

        cm=x3.coordinate_manager
        coords = cm.get_coordinates(8)
        batch_coords=coords[:,0]
        lang_feat_cast=lang_feat[batch_coords.long(),:]
        lang_sparse=ME.SparseTensor(
            features=lang_feat_cast,
            coordinate_map_key=x3.coordinate_map_key,
            coordinate_manager=cm
        )
        x3=ME.cat(x3,lang_sparse)
        x3=self.fuse_layer(x3)

        d3 = self.de3(x2, x3)
        d2 = self.de2(x1, d3)
        d1 = self.de1(input, d2)
        net = self.final_layer(d1)
        return net

class Voxel_Match(nn.Module):
    def __init__(self,in_channels=3,
                 lang_input_size=300,
                 lang_hidden_size=256,
                 use_loss=True,
                 balance_loss=True,
                 num_points=20000,
                 ):
        super().__init__()
        self.use_loss=use_loss
        self.num_points=num_points
        self.balance_loss=balance_loss
        self.UNet=UNet_fuselang(en_ch=[in_channels,32,64,128],de_ch=[128,64,32,1],lang_feat_dim=lang_hidden_size)
        self.gru = nn.GRU(input_size=lang_input_size, hidden_size=lang_hidden_size, batch_first=True)
        self.criterion=nn.MSELoss()

    def forward(self,input_dict):
        sinput=input_dict["sinput"]
        positive_mask=input_dict["positive_mask"]
        #reverse_mapping = input_dict["reverse_mapping"]
        #bcoords=input_dict["discrete_coords"]

        lang_feat=input_dict["lang_feat"]
        batch_size=lang_feat.shape[0]

        lang_feat = pack_padded_sequence(lang_feat, input_dict["lang_len"].cpu(), batch_first=True, enforce_sorted=False)
        lang_output, lang_feat = self.gru(lang_feat)  # B 256

        lang_feat = lang_feat[0]  # B 256

        heatmap=self.UNet(sinput,lang_feat)
        gather_heatmap=heatmap.slice(sinput).F
        split_list=[self.num_points for i in range(batch_size)]
        batch_heatmap=torch.split(gather_heatmap,split_list)
        batch_heatmap=torch.stack(batch_heatmap).transpose(1,2)
        target=input_dict["atten_label"].unsqueeze(1)

        output_dict={
            "atten":batch_heatmap,
            "lang_feat":lang_feat,
        }
        if self.use_loss:
            loss=self.compute_loss(batch_heatmap,target,positive_mask)
            output_dict["loss"]=loss
        return output_dict

    def compute_loss(self,output,label,positive_mask):
        '''
            loss is computed for label>0.2 and label<0.2 respectively
            for label balancing
        '''
        label=label*positive_mask[:,None,None]
        positive_ind=torch.where(label>0.2)
        positive_loss=self.criterion(output[positive_ind[0],0,positive_ind[2]],label[positive_ind[0],0,positive_ind[2]])
        negative_ind=torch.where(label<=0.2)
        negative_loss=self.criterion(output[negative_ind[0],0,negative_ind[2]],label[negative_ind[0],0,negative_ind[2]])
        loss=positive_loss+negative_loss
        loss=loss/2
        return loss