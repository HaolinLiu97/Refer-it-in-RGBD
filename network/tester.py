import torch
import os
import numpy as np
import MinkowskiEngine as ME
import datetime
import time
import pickle

def save_obj(file_path,obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def compute_Acc(IoU,result_dict):
    pred_conf=result_dict["pred_conf"].squeeze(1)
    max_conf,ind=torch.max(pred_conf,dim=1)
    max_conf_iou=IoU[torch.arange(IoU.shape[0]),ind.long()]
    Acc50=torch.sum((max_conf_iou>0.5).float())/IoU.shape[0]
    Acc25 = torch.sum((max_conf_iou > 0.25).float()) / IoU.shape[0]
    return Acc50,Acc25

def ref_tester(cfg,model_list,loss_func,test_loader,device,checkpoint):
    config = cfg.config
    if config["resume"]==True:
        print("loading checkpoint from",config["weight"])
        checkpoint.load(config["weight"])
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()

    max_Acc50=0
    iter=0
    Acc50_total = 0
    Acc25_total = 0
    eval_loss=0
    eval_loss_dict = {
        'matching_loss': 0,
        'response_loss': 0,
        'contain_loss': 0,
        'heatmap_loss': 0,
        'ref_loss': 0,
        'vote_loss': 0,
    }
    for model in model_list:
        model.eval()
    output_info={}
    with torch.no_grad():
        for batch_id, data_dict in enumerate(test_loader):
            for key in data_dict:
                if (not isinstance(data_dict[key], list)) and key!="vox_feats" and key!="vox_coords":
                    data_dict[key] = data_dict[key].cuda()
            # print(data_dict["size"])
            unique_feats = data_dict["vox_feats"]
            bcoords = ME.utils.batched_coordinates(data_dict["vox_coords"])
            sinput = ME.SparseTensor(
                unique_feats,
                bcoords,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                device=device
            )
            data_dict["sinput"] = sinput
            hm_ret_dict = model_list[0](data_dict)
            data_dict["pcd_heatmap"] = hm_ret_dict["pcd_heatmap"]
            data_dict["lang_feat"] = hm_ret_dict["lang_feat"]
            ret_dict = model_list[1](data_dict)
            total_loss, loss_dict, info_dict,IoU = loss_func(ret_dict, data_dict)
            Acc50,Acc25=compute_Acc(IoU,ret_dict)
            Acc50_total+=Acc50
            Acc25_total+=Acc25
            for key in loss_dict:
                eval_loss_dict[key]+=loss_dict[key]
            eval_loss+=total_loss

            if config['save_vis']==True:
                batch_size=data_dict["input_point_cloud"].shape[0]
                if config["data"]["dataset"] == "sunrefer":
                    unique_id_list=data_dict["image_id"]
                else:
                    unique_id_list=data_dict["scene_id"]
                object_id_list=data_dict["object_id"]
                ann_id_list=data_dict["ann_id"]
                image_id_list=data_dict["image_id"]

                pred_box_loc = ret_dict['cluster_loc'] + ret_dict['pred_partial_loc'][:, 0:3]
                pred_box_size = ret_dict['pred_partial_loc'][:, 3:6]
                pred_partial_bbox = torch.cat([pred_box_loc, pred_box_size], dim=1)

                pred_intact_loc = pred_box_loc + ret_dict['pred_intact_loc'][:, 0:3]
                pred_intact_size = ret_dict["pred_intact_loc"][:, 3:6]
                pred_intact_bbox = torch.cat([pred_intact_loc, pred_intact_size], dim=1)

                for i in range(batch_size):
                    unique_id=unique_id_list[i]
                    object_id = object_id_list[i]
                    ann_id = ann_id_list[i]
                    image_id = image_id_list[i]
                    if unique_id not in output_info:
                        output_info[unique_id]={}
                    print("processing,scene_id:%s,object id:%s,ann_id:%s,image_id:%s" % (
                    unique_id, object_id, ann_id, image_id))
                    if not output_info[unique_id].get(object_id):
                        output_info[unique_id][object_id] = []
                    pack_data = {}
                    pack_data["ann_id"] = ann_id
                    pack_data["sentence"] = data_dict['sentence'][i]
                    pack_data["image_id"] = image_id
                    pack_data["object_id"] = object_id
                    pack_data["seed_loc"] = ret_dict["seed_loc"][i].detach().cpu().numpy()
                    #pack_data["all_target_bboxes"] = data_dict["all_target_bboxes"].cpu().numpy()
                    #pack_data["target_bboxes_mask"] = data_dict["target_bboxes_mask"].cpu().numpy()
                    # pack_data["voxel_ouput"]=sparse_voxel
                    pack_data["input_point_cloud"] = data_dict["input_point_cloud"].transpose(1, 2)[
                        i].detach().cpu().numpy()
                    pack_data["vote_loc"] = ret_dict["vote_loc"][i].detach().cpu().numpy()
                    pack_data["pred_intact_box"] = pred_intact_bbox[i].detach().cpu().numpy()
                    pack_data["pred_partial_box"] = pred_partial_bbox[i].detach().cpu().numpy()
                    pack_data["vote_ind"] = ret_dict["vote_inds"][i].detach().cpu().numpy()
                    pack_data["cluster_loc"] = ret_dict["cluster_loc"][i].detach().cpu().numpy()
                    pack_data["gt_intact_bbox"] = data_dict['intact_gt_bbox'][i].detach().cpu().numpy()
                    pack_data["gt_partial_bbox"] = data_dict['partial_gt_bbox'][i].detach().cpu().numpy()
                    pack_data["pred_conf"]=ret_dict["pred_conf"][i].detach().cpu().numpy()
                    output_info[unique_id][object_id].append(pack_data)
        for scene_id in output_info:
            if os.path.exists(log_dir) == False:
                os.makedirs(log_dir)
            output_filename = os.path.join(log_dir, scene_id + "_referring.pkl")
            save_obj(output_filename, output_info[scene_id])
        avg_eval_loss=eval_loss/batch_id
        avg_Acc50=Acc50_total/batch_id
        avg_Acc25=Acc25_total/batch_id
    print("Acc50 is",avg_Acc50.item())
    print("Acc25 is",avg_Acc25.item())
    return