import torch.nn as nn
import torch
from loss.loss_utils import nn_distance,huber_loss
from loss.voxel_match_loss import voxel_match_loss

class ref_loss(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.voxel_match_loss=voxel_match_loss()
        self.cfg=cfg
        self.l2criterion=nn.MSELoss()

    def compute_iou(self,pred_box,ref_bbox):
        batch_size=pred_box.shape[0]
        N=pred_box.shape[2]

        #print(pred_box_loc,pred_box_size)

        pred_box_rb=pred_box[:,0:3]-pred_box[:,3:6]/2.0
        ref_bbox_rb=ref_bbox[:,0:3]-ref_bbox[:,3:6]/2.0

        pred_box_lt=pred_box[:,0:3]+pred_box[:,3:6]/2.0
        ref_bbox_lt=ref_bbox[:,0:3]+ref_bbox[:,3:6]/2.0

        lt=torch.min(
            pred_box_lt,
            ref_bbox_lt.unsqueeze(2).repeat(1,1,N)
        )
        rb=torch.max(
            pred_box_rb,
            ref_bbox_rb.unsqueeze(2).repeat(1,1,N)
        )

        whz=lt-rb
        whz=torch.clamp(whz,min=0)
        inter=whz[:,0,:]*whz[:,1,:]*whz[:,2,:]

        pred_box_area=pred_box[:,3,:]*pred_box[:,4,:]*pred_box[:,5,:]
        ref_box_area=ref_bbox[:,3]*ref_bbox[:,4]*ref_bbox[:,5]
        iou=inter/(pred_box_area+ref_box_area.unsqueeze(1).repeat(1,N)-inter)

        return iou
    def compute_reference_loss(self, pred_box, ref_bbox):
        batch_size = pred_box.shape[0]

        center_label = ref_bbox[:, 0:3]
        size_label = ref_bbox[:, 3:6]

        dist1, _, dist2, ind2 = nn_distance(pred_box[:,0:3].transpose(1, 2), center_label.unsqueeze(1))

        ind2 = ind2.squeeze(1)
        nn_coor_loss = torch.sum(dist2) / batch_size

        size_loss = torch.sum((pred_box[torch.arange(batch_size), 3:6, ind2] - size_label) ** 2)

        size_loss = size_loss / batch_size
        reference_loss = nn_coor_loss + size_loss

        return reference_loss

    def compute_vote_loss(self,result_dict,data_dict,loss_dict,info_dict):
        vote_loc,seed_ind=result_dict["vote_loc"],result_dict["seed_ind"]
        pseudo_vote_loc,pseudo_seed_ind=result_dict["pseudo_vote_loc"],result_dict["pseudo_seed_ind"]
        point_votes,point_votes_mask=data_dict["point_votes"],data_dict["point_votes_mask"]

        batch_size=vote_loc.shape[0]
        target_votes=point_votes[torch.arange(batch_size)[:,None],seed_ind]
        target_mask=point_votes_mask[torch.arange(batch_size)[:,None],seed_ind]
        error = torch.sum(
            (vote_loc.transpose(1, 2) * target_mask.unsqueeze(2) - target_votes * target_mask.unsqueeze(2))**2)
        #error = vote_loc.transpose(1, 2) * target_mask.unsqueeze(2)-target_votes* target_mask.unsqueeze(2)
        #vote_loss=torch.sum(huber_loss(error))/torch.sum(target_mask)
        vote_loss=error/torch.sum(target_mask)
        #vote_loss=torch.sum((vote_loc.transpose(1,2)*target_mask.unsqueeze(2)-target_votes*target_mask.unsqueeze(2))**2)/torch.sum(target_mask)

        pseudo_target_votes = point_votes[torch.arange(batch_size)[:, None], pseudo_seed_ind]
        pseudo_target_mask = point_votes_mask[torch.arange(batch_size)[:, None], pseudo_seed_ind]
        #pseudo_error = pseudo_vote_loc.transpose(1, 2) * pseudo_target_mask.unsqueeze(2) - pseudo_target_votes * pseudo_target_mask.unsqueeze(2)
        pseudo_error= torch.sum((pseudo_vote_loc.transpose(1, 2) * pseudo_target_mask.unsqueeze(2) - pseudo_target_votes * pseudo_target_mask.unsqueeze(2))**2)
        pseudo_vote_loss=pseudo_error/torch.sum(pseudo_target_mask)
        #pseudo_vote_loss=torch.sum(huber_loss(pseudo_error))/torch.sum(pseudo_target_mask)
        loss_dict["vote_loss"]=vote_loss+pseudo_vote_loss
        return loss_dict,info_dict

    def compute_matching_loss(self,result_dict,data_dict,loss_dict,info_dict):
        IoU=self.compute_iou(result_dict["pred_partial_bbox"],data_dict["partial_gt_bbox"])
        matching_loss = self.l2criterion(IoU.detach(), result_dict["pred_conf"])
        info_dict["partial_max_IoU"],_=torch.max(IoU,dim=1)
        loss_dict["matching_loss"]=matching_loss
        result_dict["partial_IoU"]=IoU
        return result_dict,loss_dict,info_dict

    def compute_response_loss(self,result_dict,data_dict,loss_dict,info_dict):
        IoU = self.compute_iou(result_dict["pred_intact_bbox"],data_dict["intact_gt_bbox"])
        max_iou, ind = torch.max(IoU, dim=1)
        pred_intact_box = result_dict["pred_intact_bbox"]
        batch_size=pred_intact_box.shape[0]
        mask = torch.zeros((pred_intact_box.shape[0])).cuda()
        valid_iou_ind = torch.where(max_iou > 0.2)
        mask[valid_iou_ind[0]] = 1
        #mask=torch.where(max_iou>0.2,torch.ones(max_iou.shape).cuda(),torch.zeros(max_iou.shape).cuda())

        nn_box = pred_intact_box[torch.arange(batch_size), :, ind]
        #print(nn_box.shape,mask.shape,data_dict["intact_gt_bbox"].shape)
        intact_response_loss = self.l2criterion(nn_box[:, 0:6]*mask.unsqueeze(1).repeat(1,6),
                                       data_dict["intact_gt_bbox"][:, 0:6]*mask.unsqueeze(1).repeat(1,6))
        #print(intact_response_loss)

        info_dict["intact_max_IoU"]=max_iou

        max_iou, ind = torch.max(result_dict["partial_IoU"], dim=1)
        pred_partial_bbox = result_dict["pred_partial_bbox"]
        mask = torch.zeros((pred_partial_bbox.shape[0])).cuda()
        valid_iou_ind = torch.where(max_iou > 0)
        mask[valid_iou_ind[0]] = 1

        nn_box = pred_partial_bbox[torch.arange(batch_size), :, ind]
        partial_response_loss = self.l2criterion(nn_box[:, 0:6]*mask.unsqueeze(1).repeat(1,6) ,
                                         data_dict["partial_gt_bbox"][:, 0:6]*mask.unsqueeze(1).repeat(1,6))
        nn_conf = result_dict["pred_conf"][torch.arange(batch_size), :, ind]
        contain_loss = self.l2criterion(nn_conf, max_iou.detach())
        loss_dict["response_loss"]=partial_response_loss+intact_response_loss
        loss_dict["contain_loss"]=contain_loss
        return loss_dict,info_dict,IoU

    def parse_result_dict(self,result_dict):
        pred_box_loc = result_dict['cluster_loc'] + result_dict['pred_partial_loc'][:, 0:3]
        pred_box_size = result_dict['pred_partial_loc'][:, 3:6]
        pred_partial_bbox = torch.cat([pred_box_loc, pred_box_size], dim=1)

        pred_intact_loc = pred_box_loc + result_dict['pred_intact_loc'][:, 0:3]
        pred_intact_size = result_dict["pred_intact_loc"][:, 3:6]
        pred_intact_bbox=torch.cat([pred_intact_loc,pred_intact_size],dim=1)
        result_dict["pred_partial_bbox"]=pred_partial_bbox
        result_dict["pred_intact_bbox"]=pred_intact_bbox
        return result_dict

    def compute_all_loss(self,result_dict,data_dict):
        '''
        result_dict contains the prediction result
        data_dict contains the GT
        '''
        result_dict=self.parse_result_dict(result_dict)
        loss_dict={}
        info_dict={}
        result_dict,loss_dict,info_dict=self.compute_matching_loss(result_dict,data_dict,loss_dict,info_dict)
        loss_dict,info_dict=self.compute_vote_loss(result_dict,data_dict,loss_dict,info_dict)
        loss_dict,info_dict,IoU=self.compute_response_loss(result_dict,data_dict,loss_dict,info_dict)

        partial_ref_loss=self.compute_reference_loss(result_dict["pred_partial_bbox"],data_dict["partial_gt_bbox"])
        intact_ref_loss=self.compute_reference_loss(result_dict["pred_intact_bbox"],data_dict["intact_gt_bbox"])
        ref_loss=partial_ref_loss+intact_ref_loss
        loss_dict["ref_loss"]=ref_loss

        voxel_match_loss=self.voxel_match_loss(result_dict["pcd_heatmap"],data_dict["heatmap_label"])
        loss_dict["heatmap_loss"]=voxel_match_loss
        return loss_dict,info_dict,IoU
    def forward(self,result_dict,data_dict):
        loss_dict,info_dict,IoU=self.compute_all_loss(result_dict,data_dict)
        total_loss=0
        for loss in loss_dict:
            loss_name=loss[0:-5]
            weight=self.cfg['loss']["w_"+loss_name]
            total_loss+=weight*loss_dict[loss]
        return total_loss,loss_dict,info_dict,IoU