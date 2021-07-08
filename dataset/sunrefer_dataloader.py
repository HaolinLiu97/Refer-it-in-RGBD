import os,sys
sys.path.append('../')
import time
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
import glob
import gzip
import pickle as p

from dataset.dataset_util import random_sampling,rotx,roty,rotz,rotate_aligned_boxes_along_axis
import math
import torch
import cv2

MEAN_COLOR_RGB= np.array([109.8, 97.2, 83.8])

class sunrefer_dataset(Dataset):
    def __init__(self,cfg,isTrain=True):
        self.cfg=cfg
        self.sigma = cfg['data']['sigma']
        self.debug = cfg['data']['debug']
        self.voxel_size = cfg['data']['voxel_size']
        self.use_color = cfg['data']['use_color']
        self.num_points = cfg['data']['num_points']
        self.use_aug = cfg['data']['use_aug']
        self.isTrain = isTrain

        if self.isTrain:
            self.split_file = cfg['data']['train_path']
        else:
            self.split_file = cfg['data']['val_path']
            self.use_aug=False
        if self.split_file.find(".pkl")>=0:
            with open(self.split_file,'rb') as f:
                self.sunrgbd=p.load(f)
        else:
            with open(self.split_file, 'r') as f:
                self.sunrgbd = json.load(f)

        self.__load_data()
    def __len__(self):
        return len(self.sunrgbd)

    def convert_orientedbbox2AABB(self,all_bboxes):
        c_x=all_bboxes[:,0]
        c_y=all_bboxes[:,1]
        c_z=all_bboxes[:,2]
        s_x=all_bboxes[:,3]
        s_y=all_bboxes[:,4]
        s_z=all_bboxes[:,5]
        angle=all_bboxes[:,6]
        orientation=np.concatenate([np.cos(angle)[:,np.newaxis],
                                    -np.sin(angle)[:,np.newaxis]],axis=1)
        ori1=orientation
        ori2=np.ones(ori1.shape)
        ori2=ori2-np.sum(ori1*ori2,axis=1)[:,np.newaxis]*ori1
        ori2=ori2/np.linalg.norm(ori2,axis=1)[:,np.newaxis]
        ori1=ori1*s_x[:,np.newaxis]
        ori2=ori2*s_y[:,np.newaxis]
        verts = np.array([[c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2]])
        verts=verts.transpose(2,0,1)
        verts[:,0,0:2] = verts[:,0, 0:2] - ori2 / 2 - ori1 / 2
        verts[:,1, 0:2] = verts[:,1, 0:2] - ori2 / 2 - ori1 / 2
        verts[:,2, 0:2] = verts[:,2, 0:2] - ori2 / 2 + ori1 / 2
        verts[:,3, 0:2] = verts[:,3, 0:2] - ori2 / 2 + ori1 / 2
        verts[:,4, 0:2] = verts[:,4, 0:2] + ori2 / 2 - ori1 / 2
        verts[:,5, 0:2] = verts[:,5, 0:2] + ori2 / 2 - ori1 / 2
        verts[:,6, 0:2] = verts[:,6, 0:2] + ori2 / 2 + ori1 / 2
        verts[:,7, 0:2] = verts[:,7, 0:2] + ori2 / 2 + ori1 / 2
        x_min=np.min(verts[:,:,0],axis=1)
        x_max=np.max(verts[:,:,0],axis=1)
        y_min=np.min(verts[:,:,1],axis=1)
        y_max=np.max(verts[:,:,1],axis=1)
        z_min=np.min(verts[:,:,2],axis=1)
        z_max=np.max(verts[:,:,2],axis=1)
        cx=(x_min+x_max)/2
        cy=(y_min+y_max)/2
        cz=(z_min+z_max)/2
        sx=x_max-x_min
        sy=y_max-y_min
        sz=z_max-z_min
        AABB_bbox=np.concatenate([cx[:,np.newaxis],cy[:,np.newaxis],cz[:,np.newaxis],
                                  sx[:,np.newaxis],sy[:,np.newaxis],sz[:,np.newaxis]],axis=1)

        return AABB_bbox

    def __getitem__(self,idx):
        image_id=str(self.sunrgbd[idx]['image_id'])
        sentence=self.sunrgbd[idx]['sentence']
        object_id=self.sunrgbd[idx]['object_id']
        ann_id=self.sunrgbd[idx]["ann_id"]

        #-----------------------------load language feature-----------------------
        lang_feat=self.lang[image_id][object_id][ann_id]
        lang_len = len(self.sunrgbd[idx]["tokens"])
        lang_len = lang_len if lang_len <= self.cfg['data']['MAX_DES_LEN'] else self.cfg['data']['MAX_DES_LEN']

        #-----------------------------load point cloud-----------------------------

        point_cloud=self.image_data[image_id]["point_cloud"].copy()
        bbox=self.image_data[image_id]['bbox'].copy()
        bbox[:,3:6]=bbox[:,3:6]*2

        if not self.use_color:
            point_cloud=point_cloud[:,0:3]
        else:
            point_cloud=point_cloud[:,0:6]
            point_cloud[:,3:]=(point_cloud[:,3:]*255-MEAN_COLOR_RGB)/128#normalize the point cloud between -1 to 1#

        #print(bbox.shape)
        #---------------------------------LABELS-------------------------------
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        target_bboxes = np.zeros((self.cfg['data']['MAX_NUM_OBJ'], 7))
        target_bboxes_mask = np.zeros((self.cfg['data']['MAX_NUM_OBJ']))
        num_bbox=bbox.shape[0] if bbox.shape[0]<self.cfg['data']['MAX_NUM_OBJ'] else self.cfg['data']['MAX_NUM_OBJ']
        target_bboxes_mask[0:num_bbox]=1
        target_bboxes[0:num_bbox, :] = bbox[:num_bbox, :]  # xyzwhl

        #------------------------------votes-------------------------------------
        point_votes=self.image_data[image_id]["votes"].copy()
        #print(point_votes.shape)
        point_votes_end=point_votes[choices,1:10]
        #print(point_votes_end[:,0:3].shape,point_cloud[:,0:3].shape)
        point_votes_end[:,0:3] += point_cloud[:,0:3]
        point_votes_end[:, 3:6] += point_cloud[:, 0:3]
        point_votes_end[:, 6:9] += point_cloud[:, 0:3]
        point_votes_mask=point_votes[choices,0]

        # ------------------------------- DATA AUGMENTATION --------------------------
        if self.use_aug:
            if np.random.random()>0.5:
                # Flipping along the YZ plane
                point_cloud[:,0]= -1*point_cloud[:,0]
                target_bboxes[:,0]=-1*target_bboxes[:,0]
                target_bboxes[:, 6] = np.pi - target_bboxes[:, 6]
                point_votes_end[:, 0] = -point_votes_end[:, 0]
                point_votes_end[:, 3] = -point_votes_end[:, 3]
                point_votes_end[:, 6] = -point_votes_end[:, 6]

            # Rotation along Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ 30 degree
            rot_mat = rotz(rot_angle)
            target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
            target_bboxes[:, 6] -= rot_angle
            point_cloud[:, 0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            point_votes_end[:, 0:3] = np.dot(point_votes_end[:, 0:3], np.transpose(rot_mat))
            point_votes_end[:, 3:6] = np.dot(point_votes_end[:, 3:6], np.transpose(rot_mat))
            point_votes_end[:, 6:9] = np.dot(point_votes_end[:, 6:9], np.transpose(rot_mat))


            point_cloud,target_bboxes,point_votes_end=self._translate(point_cloud,target_bboxes,point_votes_end)

        # --------------------------generate partial bounding box for partial scan----------------------
        AABB_target_bboxes=self.convert_orientedbbox2AABB(target_bboxes)
        ref_box_label = np.zeros(self.cfg['data']['MAX_NUM_OBJ'])
        ref_box_label[int(object_id)]=1
        ref_bbox=AABB_target_bboxes[int(object_id)]
        #orientated_ref_bbox=target_bboxes[object_id]

        sigma = 0.5

        dist = (point_cloud[:, 0:3] - ref_bbox[0:3]) ** 2
        x_std = ref_bbox[3]
        y_std = ref_bbox[4]
        z_std = ref_bbox[5]
        dist = dist / np.array([x_std ** 2, y_std ** 2, z_std ** 2])
        # print(dist.shape)
        dist = np.sqrt(np.sum(dist, axis=1))
        gaussian_kernel = np.exp(-dist / 2 / sigma ** 2)
        atten_label = gaussian_kernel
        '''discrete_coords, unique_feats= ME.utils.sparse_quantize(
            coords=point_cloud[:,0:3],
            feats=point_cloud[:,3:],
            quantization_size=0.05)'''
        xyz = point_cloud[:, 0:3].copy()
        feats = point_cloud[:, 3:6].copy()
        if self.use_aug:
            # Rotation along Z-axis
            rot_angle = (np.random.random() * np.pi/18) - np.pi / 36  # -5 ~ 5 degree
            rot_mat = rotz(rot_angle)
            xyz = np.dot(xyz, np.transpose(rot_mat))

            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ 5 degree
            rot_mat = rotx(rot_angle)
            xyz = np.dot(xyz, np.transpose(rot_mat))

            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ 5 degree
            rot_mat = roty(rot_angle)
            xyz = np.dot(xyz, np.transpose(rot_mat))

            alpha = np.random.random() * 0.4 + 0.8  # constrast control from 0.8~1.2
            beta = np.random.random() * 0.2 - 0.1  # brightness control from -0.1~0.1
            random_R_control = np.random.random() * 0.2 + 0.9
            random_G_control = np.random.random() * 0.2 + 0.9
            random_B_control = np.random.random() * 0.2 + 0.9
            feats[:, 0] = feats[:, 0] * random_R_control
            feats[:, 1] = feats[:, 1] * random_G_control
            feats[:, 2] = feats[:, 2] * random_B_control
            feats = feats * alpha + beta

        coords = xyz / self.voxel_size

        batch={}
        batch["vox_coords"] = coords
        batch["ann_id"]=ann_id
        #batch["orientated_ref_bbox"]=orientated_ref_bbox
        batch["vox_feats"] = feats
        batch["heatmap_label"] = atten_label
        batch["partial_gt_bbox"]=ref_bbox
        batch["intact_gt_bbox"]=ref_bbox
        batch["input_point_cloud"]=point_cloud.T
        batch["lang_feat"]=lang_feat
        batch['lang_len']=lang_len
        batch['object_id']=str(object_id)

        batch['image_id']=image_id
        batch['point_votes']=point_votes_end
        batch['point_votes_mask']=point_votes_mask
        batch['sentence']=sentence

        return batch

    def __load_data(self):
        start_t = time.time()
        self.lang=self._tranform_des()
        print(list(set([data["image_id"] for data in self.sunrgbd])))
        self.image_list=sorted(list(set([data["image_id"] for data in self.sunrgbd])))
        if self.debug==True:
            self.image_list=self.image_list[0:700]

        self.image_data={}
        for image_id in self.image_list:
            print("loading",image_id)
            self.image_data[image_id]={}
            self.image_data[image_id]['bbox']=np.array(np.load(os.path.join(self.cfg['data']['data_path'],image_id+"_bbox.npy")))
            point_cloud=np.load(os.path.join(self.cfg['data']['data_path'],image_id+"_pc.npz"))["pc"]
            self.image_data[image_id]["point_cloud"]=point_cloud
            votes=np.load(os.path.join(self.cfg['data']['data_path'],image_id+"_votes.npz"))["point_votes"]
            self.image_data[image_id]["votes"]=votes


            #print(self.scene_data[scene_id]['object_id_to_label_id'])
        end_t = time.time()
        print("it takes %f seconds to load the whole dataset"%(end_t-start_t))
    def _tranform_des(self):
        with open(os.path.join('./data','glove.p'), "rb") as f:
            glove = pickle.load(f)

        lang = {}
        for data in self.sunrgbd:
            image_id = data["image_id"]
            object_id = data["object_id"]
            ann_id=data["ann_id"]
            tokens=data["tokens"]

            if image_id not in lang:
                lang[image_id] = {}

            if object_id not in lang[image_id]:
                lang[image_id][object_id] = {}

            # tokenize the description
            embeddings = np.zeros((self.cfg['data']['MAX_DES_LEN'], 300))
            #print("one sentence")
            for token_id in range(self.cfg['data']['MAX_DES_LEN']):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        #print(token)
                        embeddings[token_id] = glove[token]
                    else:
                        #print("invalid token",token)
                        embeddings[token_id] = glove["unk"]

            # store
            lang[image_id][object_id][ann_id] = embeddings

        return lang


    def _translate(self, point_set, bbox,point_votes_end):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor
        point_votes_end[:,0:3]+=factor
        point_votes_end[:,3:6]+=factor
        point_votes_end[:,6:9]+=factor
        return point_set, bbox,point_votes_end
