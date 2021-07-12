import torch
import torch.nn as nn
from third_party.pointnet2.pointnet2_modules import PointnetLFPModuleMSG,PointnetSAModuleVotes
import network.modules as modules
from weighted_FPS.weighted_FPS_utils import weighted_furthest_point_sample

class RGBD_RefNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.is_train=self.cfg['isTrain']
        self.alpha=self.cfg['model']['alpha']
        self.t=self.cfg['model']['t']
        self.max_nseed = self.cfg['model']['max_nseed']
        self.pcd_in_channels=self.cfg['model']['pcd_in_channels']
        self.pcd_hidden_size=self.cfg['model']['pcd_hidden_size']
        self.lang_hidden_size=self.cfg['model']['lang_hidden_size']
        self.encoder = modules.keypoint_encoder(in_channels=self.pcd_in_channels, output_channels=self.pcd_hidden_size,
                                       npoint=[2048, 1024, 512], radius=[0.1, 0.2, 0.4],
                                       nsample=[64, 32, 16],
                                       mlp=[[3, 64, 64, 128], [128, 128, 128, 256], [256, 128, 128, 256]],
                                       ret_first=True)
        self.seed_feat_aggreagtion = PointnetLFPModuleMSG(mlps=[[256, 256, 256]], radii=[0.4], nsamples=[16],
                                               post_mlp=[256 + 3, 256])
        self.intact_bbox_grouper = PointnetLFPModuleMSG(mlps=[[512, 512, 256, 128]], radii=[0.4], nsamples=[32],
                                                        post_mlp=[256, 128])
        self.vgen = modules.VotingModule(vote_factor=1, seed_feature_dim=512)
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=32,
            radius=0.3,
            nsample=16,
            mlp=[512, 256, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        self.fuse_text_seed = nn.Sequential(
            nn.Conv1d(in_channels=self.pcd_hidden_size + self.lang_hidden_size, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
        )

        self.res_mlp = nn.Sequential(
            modules.res_mlp(128, 2, use_2d=False),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.res_intact = nn.Sequential(
            modules.res_mlp(128, 2, use_2d=False),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.ReLU()
        )
        self.partial_loc_regress = nn.Sequential(
            nn.Conv1d(128, 6, kernel_size=1)
        )
        self.intact_loc_regress = nn.Sequential(
            nn.Conv1d(128, 6, kernel_size=1)
        )
        self.conf_regress = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_dict):
        input_point_cloud = input_dict["input_point_cloud"]
        lang_feat = input_dict["lang_feat"]
        pcd_heatmap = input_dict["pcd_heatmap"]
        pcd_heatmap = torch.clamp(pcd_heatmap, min=0, max=1)

        batch_size=input_point_cloud.shape[0]

        '''
        512 keypoitns are generated
        '''
        key_xyz, pointwise_feat, coor_dict = self.encoder(input_point_cloud)
        ind_512=coor_dict["ind_512"]

        '''
        conduct weighted FPS to select 256 seed points
        '''
        xyzh=torch.cat([input_point_cloud[:,0:3,:].transpose(1,2),pcd_heatmap[:,:,None]],dim=2)
        point_ind=weighted_furthest_point_sample(xyzh,self.max_nseed,self.alpha,self.t).long()
        seed_xyz = input_point_cloud[:, 0:3, :].transpose(1, 2)[torch.arange(batch_size)[:, None], point_ind, 0:3]


        '''seed points feature aggregation'''
        seed_feat = self.seed_feat_aggreagtion(seed_xyz.contiguous(), key_xyz, seed_xyz.transpose(1, 2), pointwise_feat)
        cat_feat = torch.cat([seed_feat, lang_feat.unsqueeze(2).repeat(1, 1, seed_feat.shape[2])], dim=1)
        fuse_feat = self.fuse_text_seed(cat_feat)
        vote_xyz, features = self.vgen(seed_xyz, fuse_feat)

        '''conduct voting'''
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        xyz, features, fps_inds = self.vote_aggregation(vote_xyz, features)
        vote_inds = point_ind[torch.arange(batch_size)[:, None], fps_inds.long()] #index for the vote in point cloud

        '''use residual mlp to process the proposal features'''
        feat = self.res_mlp(features)
        partial_loc = self.partial_loc_regress(feat)
        conf = self.conf_regress(feat)

        '''proposal refinement'''
        partial_center = partial_loc[:, 0:3].transpose(1, 2) + xyz

        intact_feat = self.intact_bbox_grouper(partial_center.contiguous(), seed_xyz.contiguous(), feat, fuse_feat)
        intact_feat = self.res_intact(intact_feat)

        intact_loc = self.intact_loc_regress(intact_feat)

        '''get the heat value per vote'''
        vote_heatmap = pcd_heatmap[torch.arange(batch_size)[:, None], vote_inds.long()]
        ret_dict = {
            "cluster_loc": xyz.transpose(1, 2),
            "vote_heatmap": vote_heatmap,
            "vote_loc": vote_xyz.transpose(1, 2),
            "seed_loc": seed_xyz.transpose(1, 2),
            "seed_ind": point_ind,
            "vote_inds": vote_inds,
            "pred_partial_loc": partial_loc,
            "pred_intact_loc": intact_loc,
            "pred_conf": conf,
            "pcd_heatmap": pcd_heatmap
        }
        '''pseudo seed points are generated by FPS, it is used for only the supervision
                   self aggregation'''
        pseudo_seed_feat = self.seed_feat_aggreagtion(key_xyz.contiguous(), key_xyz, key_xyz.transpose(1, 2),
                                                      pointwise_feat)
        pseudo_cat_feat = torch.cat(
            [pseudo_seed_feat, lang_feat.unsqueeze(2).repeat(1, 1, pseudo_seed_feat.shape[2])],
            dim=1)
        pseudo_fuse_feat = self.fuse_text_seed(pseudo_cat_feat)
        pseudo_vote_xyz, _ = self.vgen(key_xyz, pseudo_fuse_feat)
        ret_dict["pseudo_seed_ind"]= ind_512
        ret_dict['pseudo_vote_loc']=pseudo_vote_xyz.transpose(1, 2)
        return ret_dict