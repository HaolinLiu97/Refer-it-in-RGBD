import os
import json
import pickle as p

sunrefer_path="../data/SUNREFER_v2.json"
sunspot_path="../data/SUNSPOT_3D.pkl"
save_train_path="../data/sunrefer_singleRGBD/SUNREFER_train.pkl"
save_val_path="../data/sunrefer_singleRGBD/SUNREFER_val.pkl"

with open(sunrefer_path,'r') as f:
    sunrefer_content=json.load(f)

with open(sunspot_path,'rb') as f:
    sunspot_content=p.load(f)


train_content=[]
val_content=[]
for item in sunrefer_content:
    item["image_id"]=item["image_id"][0:6]
    try:
        int(item["object_id"])
    except:
        #print(item)
        continue
    if int(item["image_id"])<3000 and len(item['sentence'])>1:
        val_content.append(item)
    elif int(item["image_id"])>=3000 and len(item['sentence'])>1:
        train_content.append(item)
print("----------------------------")
for item in sunspot_content:
    item["image_id"] = item["image_id"][0:6]
    try:
        int(item["object_id"])
    except:
        #print(item)
        continue
    if int(item["image_id"])<3000 and len(item['sentence'])>1:
        val_content.append(item)
    elif int(item["image_id"])>=3000 and len(item['sentence'])>1:
        train_content.append(item)

print(len(train_content))
print(len(val_content))

with open(save_train_path,'wb') as f:
    p.dump(train_content,f)
with open(save_val_path,'wb') as f:
    p.dump(val_content,f)
