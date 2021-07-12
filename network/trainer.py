import torch
from tensorboardX import SummaryWriter
import os
import numpy as np
import MinkowskiEngine as ME
import datetime
import time
import pickle as p

def compute_Acc(IoU,result_dict):
    pred_conf=result_dict["pred_conf"].squeeze(1)
    max_conf,ind=torch.max(pred_conf,dim=1)
    max_conf_iou=IoU[torch.arange(IoU.shape[0]),ind.long()]
    Acc50=torch.sum((max_conf_iou>0.5).float())/IoU.shape[0]
    Acc25 = torch.sum((max_conf_iou > 0.25).float()) / IoU.shape[0]
    return Acc50,Acc25

def voxel_match_trainer(cfg,model,loss_func,optimizer,scheduler,train_loader,test_loader,device,checkpoint):
    start_t=time.time()
    config=cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    model.train()

    min_eval_loss=1000
    for e in range(0,config['other']['nepoch']):
        print("Switch Phase to Train")
        model.train()
        for batch_id, data_dict in enumerate(train_loader):
            optimizer.zero_grad()
            for key in data_dict:
                if (not isinstance(data_dict[key],list)) and key!="vox_feats" and key!="vox_coords":
                    data_dict[key]=data_dict[key].cuda()
            # print(data_dict["size"])
            unique_feats = data_dict["vox_feats"].float()
            bcoords = ME.utils.batched_coordinates(data_dict["vox_coords"]).float()
            sinput = ME.SparseTensor(
                unique_feats,
                bcoords,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                device=device
            )
            data_dict["sinput"] = sinput
            ret_dict = model(data_dict)

            loss = loss_func(ret_dict["pcd_heatmap"],data_dict["heatmap_label"])
            total_loss = loss

            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_loader),
                "total_loss",
                total_loss.item()
            )
            print(msg)
            total_loss.backward()
            optimizer.step()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar('lr', current_lr, iter)
            tb_logger.add_scalar('loss', total_loss.item(), iter)

            '''
            visualize training data into pickle form
            '''
            if iter % config['visualization']["model_vis_interval"] == 0:
                feats_batch = data_dict["vox_feats"].detach().cpu().numpy()  # color
                bcoords = bcoords.detach().cpu().numpy()
                atten = ret_dict["pcd_heatmap"].detach().cpu().numpy()
                label = data_dict["heatmap_label"].detach().cpu().numpy()
                chosen_ind = np.where(bcoords[:, 0] == 0)[0]
                chosen_feat = feats_batch[chosen_ind, 0:3]  # color
                chosen_coords = bcoords[chosen_ind, 1:4]  # xyz
                chosen_atten = atten[0, :, np.newaxis]  # atten
                chosen_label = label[0, :, np.newaxis]
                sparse_voxel = np.concatenate([chosen_coords, chosen_feat, chosen_atten, chosen_label], axis=1)
                unique_id=None
                if cfg.config['data']['dataset']=="scanrefer-singleRGBD":
                    unique_id=data_dict["scene_id"][0]
                elif cfg.config['data']['dataset']=="sunrefer":
                    unique_id=data_dict["image_id"][0]
                voxel_dict = {
                    "voxel_output": sparse_voxel,
                    "scene_id": unique_id,
                    "sentence": data_dict["sentence"][0],
                }

                save_filename = "voxel_output_%d.pkl" % (iter)
                save_path = os.path.join(log_dir, save_filename)
                with open(save_path, "wb") as f:
                    p.dump(voxel_dict, f)
            if iter % config['other']['clean_cache_interval'] == 0:
                torch.cuda.empty_cache()

            iter += 1
        scheduler.step()
        '''conduct test on test loader'''
        print("Switch Phase to Test")
        model.eval()
        eval_loss=0
        with torch.no_grad():
            for batch_id, data_dict in enumerate(test_loader):
                optimizer.zero_grad()
                for key in data_dict:
                    if (not isinstance(data_dict[key], list)) and key != "vox_feats" and key!="vox_coords":
                        data_dict[key] = data_dict[key].cuda()
                # print(data_dict["size"])
                unique_feats = data_dict["vox_feats"].float()
                bcoords = ME.utils.batched_coordinates(data_dict["vox_coords"]).float()
                sinput = ME.SparseTensor(
                    unique_feats,
                    bcoords,
                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                    device=device
                )
                data_dict["sinput"] = sinput
                ret_dict = model(data_dict)

                loss = loss_func(ret_dict["pcd_heatmap"],data_dict["heatmap_label"])

                total_loss=loss
                msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                    str(datetime.timedelta(seconds=round(time.time() - start_t))),
                    "epoch",
                    e,
                    batch_id + 1,
                    len(test_loader),
                    "test_loss",
                    total_loss.item()
                )

                print(msg)
                eval_loss+=loss.item()
            avg_eval_loss=eval_loss/(batch_id+1)
        print("eval_loss is",avg_eval_loss)
        tb_logger.add_scalar('eval_loss', avg_eval_loss, e)
        checkpoint.register_modules(epoch=e, min_loss=avg_eval_loss)
        if avg_eval_loss<min_eval_loss:
            checkpoint.save('best')
            min_eval_loss=avg_eval_loss
        else:
            checkpoint.save("latest")
    return

def ref_trainer(cfg,model_list,loss_func,optimizer_list,scheduler_list,train_loader,test_loader,device,checkpoint):
    config = cfg.config
    if config["resume"]==True:
        print("loading checkpoint from",config["weight"])
        checkpoint.load(config["weight"])
    start_epoch = scheduler_list[0].last_epoch
    start_t=time.time()
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])
    if os.path.exists(log_dir) == False:
        os.makedirs(log_dir)
    cfg.write_config()
    tb_logger = SummaryWriter(log_dir)

    max_Acc50=0
    iter = train_loader.__len__() * start_epoch
    Acc50_total = 0
    Acc25_total = 0
    for e in range(start_epoch,config['other']['nepoch']):
        print("Switch Phase to Train")
        acc_50_avg=0
        for model in model_list:
            model.train()
        for batch_id, data_dict in enumerate(train_loader):
            for optimizer in optimizer_list:
                optimizer.zero_grad()
            for key in data_dict:
                if (not isinstance(data_dict[key],list)) and key!="vox_feats" and key!="vox_coords":
                    data_dict[key]=data_dict[key].cuda()
            # print(data_dict["size"])
            unique_feats = data_dict["vox_feats"].float()
            bcoords = ME.utils.batched_coordinates(data_dict["vox_coords"]).float()
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
            total_loss,loss_dict,info_dict,IoU=loss_func(ret_dict,data_dict)
            Acc50, Acc25 = compute_Acc(IoU, ret_dict)
            Acc50_total+=Acc50
            Acc25_total+=Acc25

            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_loader),
                "total_loss",
                total_loss.item()
            )
            print(msg)
            total_loss.backward()
            for optimizer in optimizer_list:
                optimizer.step()
            current_lr = optimizer_list[0].state_dict()['param_groups'][0]['lr']
            tb_logger.add_scalar('lr', current_lr, iter)
            tb_logger.add_scalar('train/total_loss', total_loss.item(), iter)
            for loss in loss_dict:
                tb_logger.add_scalar("train/"+loss, loss_dict[loss].item(), iter)
            for info in info_dict:
                tb_logger.add_scalar("train/"+info,torch.mean(info_dict[info]).item(),iter)

            '''
            visualize training data into pickle form
            '''

            if iter % config['visualization']["model_vis_interval"] == 0:
                feats_batch = data_dict["vox_feats"].detach().cpu().numpy()  # color
                bcoords = bcoords.detach().cpu().numpy()
                atten = ret_dict["pcd_heatmap"].detach().cpu().numpy()
                label = data_dict["heatmap_label"].detach().cpu().numpy()
                chosen_ind = np.where(bcoords[:, 0] == 0)[0]
                chosen_feat = feats_batch[chosen_ind, 0:3]  # color
                chosen_coords = bcoords[chosen_ind, 1:4]  # xyz
                chosen_atten = atten[0, :, np.newaxis]  # atten
                chosen_label = label[0, :, np.newaxis]
                print(chosen_coords.shape, chosen_feat.shape, chosen_atten.shape, chosen_label.shape)
                sparse_voxel = np.concatenate([chosen_coords, chosen_feat, chosen_atten, chosen_label], axis=1)
                if config['data']['dataset']=="sunrefer":
                    unique_id=data_dict["image_id"][0]
                else:
                    unique_id=data_dict["scene_id"][0]
                save_dict = {
                    "voxel_output": sparse_voxel,
                    "unique_id": unique_id,
                    "sentence": data_dict["sentence"][0],
                }
                save_dict["input_point_cloud"] = data_dict["input_point_cloud"][0].detach().cpu().numpy()
                save_dict["seed_ind"] = ret_dict["seed_ind"][0].detach().cpu().numpy()
                save_dict["pseudo_seed_ind"] = ret_dict["pseudo_seed_ind"][0].detach().cpu().numpy()
                save_dict["point_votes"] = data_dict["point_votes"][0].detach().cpu().numpy()
                save_dict["point_votes_mask"] = data_dict["point_votes_mask"][0].detach().cpu().numpy()
                save_dict["vote_loc"] = ret_dict["vote_loc"][0].detach().cpu().numpy()
                save_dict["pseudo_vote_loc"] = ret_dict["pseudo_vote_loc"][0].detach().cpu().numpy()

                save_filename = "data_batch_%d.pkl" % (iter)
                save_path = os.path.join(log_dir, save_filename)
                with open(save_path, "wb") as f:
                    p.dump(save_dict, f)

            if iter % config['other']['clean_cache_interval'] == 0:
                torch.cuda.empty_cache()

            iter += 1
        train_Acc50=Acc50_total/batch_id
        train_Acc25=Acc25_total/batch_id
        tb_logger.add_scalar("train/Acc50", train_Acc50.item(), e)
        tb_logger.add_scalar("train/Acc25", train_Acc25.item(),e)
        for scheduler in scheduler_list:
            scheduler.step()
        '''conduct test on test loader'''
        print("Switch Phase to Test")
        for model in model_list:
            model.eval()
        eval_loss_dict={
            'matching_loss': 0,
            'response_loss': 0,
            'contain_loss': 0,
            'heatmap_loss': 0,
            'ref_loss': 0,
            'vote_loss': 0,
        }
        eval_loss = 0
        Acc50_total=0
        Acc25_total=0
        with torch.no_grad():
            for batch_id, data_dict in enumerate(test_loader):
                for key in data_dict:
                    if (not isinstance(data_dict[key], list)) and key!="vox_feats" and key!="vox_coords":
                        data_dict[key] = data_dict[key].cuda()
                # print(data_dict["size"])
                unique_feats = data_dict["vox_feats"].float()
                bcoords = ME.utils.batched_coordinates(data_dict["vox_coords"]).float()
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
                Acc50_total+=Acc50.item()
                Acc25_total+=Acc25.item()
                for key in loss_dict:
                    eval_loss_dict[key]+=loss_dict[key].item()
                eval_loss+=total_loss.item()
            avg_eval_loss=eval_loss/batch_id
            avg_Acc50=Acc50_total/batch_id
            avg_Acc25=Acc25_total/batch_id
            tb_logger.add_scalar('Eval/Acc50', avg_Acc50, e)
            tb_logger.add_scalar('Eval/Acc25', avg_Acc25, e)
            for key in eval_loss_dict:
                eval_loss_dict[key]=eval_loss_dict[key]/batch_id
                tb_logger.add_scalar('Eval/'+key, eval_loss_dict[key], e)
        tb_logger.add_scalar("Eval/eval_loss",avg_eval_loss,e)
        model_save_dir = os.path.join(config['other']['model_save_dir'], config['exp_name'])
        checkpoint.register_modules(epoch=e, min_loss=avg_eval_loss)
        if os.path.exists(model_save_dir)==False:
            os.makedirs(model_save_dir)
        if avg_Acc50>max_Acc50:
            max_Acc50=avg_Acc50
            checkpoint.save("best")
        else:
            checkpoint.save("latest")
    return