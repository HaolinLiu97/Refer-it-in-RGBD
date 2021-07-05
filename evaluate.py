import numpy as np
import os
import pickle as p
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str,required=True)
args = parser.parse_args()


def compute_iou(pred_box, ref_bbox):

    N=pred_box.shape[0]

    pred_box_rb = pred_box[:,0:3] - pred_box[:,3:6] / 2.0
    ref_bbox_rb = ref_bbox[:,0:3] - ref_bbox[:,3:6] / 2.0

    pred_box_lt = pred_box[:,0:3] + pred_box[:,3:6] / 2.0
    ref_bbox_lt = ref_bbox[:,0:3] + ref_bbox[:,3:6] / 2.0

    lt = np.min(np.concatenate([pred_box_lt[:,:,np.newaxis],np.repeat(ref_bbox_lt[:,:,np.newaxis],N,axis=0)],axis=2),axis=2)
    rb = np.max(np.concatenate([pred_box_rb[:,:,np.newaxis],np.repeat(ref_bbox_rb[:,:,np.newaxis],N,axis=0)],axis=2),axis=2)
    whz = lt - rb
    whz[whz < 0] = 0
    inter = whz[:, 0] * whz[:, 1] * whz[:, 2]

    pred_box_area = pred_box[:, 3] * pred_box[:, 4] * pred_box[:, 5]
    ref_box_area = ref_bbox[:, 3] * ref_bbox[:, 4] * ref_bbox[:, 5]

    # print(pred_box_area.shape,inter.shape,ref_box_area.shape)
    iou = inter / (pred_box_area + np.repeat(ref_box_area,N,axis=0) - inter)
    # print(iou)
    return iou

result_dir=args.result_dir
k=5

success_count_iou25=0
success_count_iou50=0
Rat2_count=0
Rat5_count=0
Rat10_count=0
Rat20_count=0
total_count=0
Max_IoU=0

success_count=0
iou_sum=0

par_success_count_iou25=0
par_success_count_iou50=0
par_Rat2_count=0
par_Rat5_count=0
par_Rat10_count=0
par_Rat20_count=0
par_Max_IoU=0
scan_list=os.listdir(result_dir)
#print(scan_list)
#scan_list=[scan_list[0]]
for scan_file in scan_list:
    scan_output_file=os.path.join(result_dir,scan_file)
    scan=scan_file[:12]
    #print(scan)
    with open(scan_output_file,"rb") as f:
        output_content=p.load(f)
    object_id_list=list(output_content.keys())
    for object_id in object_id_list:
        for object_data in output_content[object_id]:
            prediction=object_data["pred_intact_box"].T
            gt=object_data["gt_intact_bbox"].T
            partial_pred=object_data["pred_partial_box"]
            partial_gt=object_data["gt_partial_bbox"].T
            #prediction=partial_pred[:,0:6]
            #gt=partial_gt
            #print(prediction.shape)
            bbox=prediction[:,0:6]
            #print(prediction.shape)
            partial_bbox=partial_pred[:,0:6]
            confidence=object_data["output"][:,6]
            sort_id=np.argsort(-confidence)
            topk_id=sort_id[:20]
            topk_bbox=bbox[topk_id]
            topk_partial_bbox=partial_bbox[topk_id]

            target_bbox=gt[np.newaxis,:]
            target_partial_bbox=partial_gt[np.newaxis,:]
            #print(target_bbox.shape)

            iou=compute_iou(topk_bbox,target_bbox)
            par_iou=compute_iou(topk_partial_bbox,target_partial_bbox)
            par_Max_IoU+=np.max(par_iou)
            Max_IoU+=np.max(iou)
            if iou[0]>0.25:
                success_count_iou25+=1
            if iou[0]>0.5:
                success_count_iou50+=1
                iou_sum+=iou[0]
                success_count+=1
            if np.max(iou[0:2])>0.5:
                Rat2_count+=1
            if np.max(iou[0:5])>0.5:
                Rat5_count+=1
            if np.max(iou[0:10])>0.5:
                Rat10_count+=1
            if np.max(iou[0:20])>0.5:
                Rat20_count+=1

            if par_iou[0]>0.25:
                par_success_count_iou25+=1
            if par_iou[0]>0.5:
                par_success_count_iou50+=1
            if np.max(par_iou[0:2])>0.5:
                par_Rat2_count+=1
            if np.max(par_iou[0:5])>0.5:
                par_Rat5_count+=1
            if np.max(par_iou[0:10])>0.5:
                par_Rat10_count+=1
            if np.max(par_iou[0:20])>0.5:
                par_Rat20_count+=1
            total_count+=1
print("---------------intact_bbox--------------------------")
print("IoU25_success_rate:",success_count_iou25/total_count)
print("IoU50_success_rate:", success_count_iou50 / total_count)
print("R@2:", Rat2_count / total_count)
print("R@5:", Rat5_count / total_count)
print("R@10:", Rat10_count / total_count)
print("R@20", Rat20_count / total_count)
print("Max_IoU",Max_IoU/total_count)
print("success mean IoU",iou_sum/success_count)
print("---------------partial_bbox--------------------------")
print("IoU25_success_rate:",par_success_count_iou25/total_count)
print("IoU50_success_rate:", par_success_count_iou50 / total_count)
print("R@2:", Rat2_count / total_count)
print("R@5:", par_Rat5_count / total_count)
print("R@10:", par_Rat10_count / total_count)
print("R@20", par_Rat20_count / total_count)
print("Max_IoU",par_Max_IoU/total_count)



