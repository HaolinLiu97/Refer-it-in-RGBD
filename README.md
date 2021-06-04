# Refer-it-in-RGBD
<p align="center"><img src="docs/teaser.png" width="300px"/><img src="docs/projectpage.gif" width="330px"></p></br>
This is the repository of our paper 'Refer-it-in-RGBD: A Bottom-up Approach for 3D Visual Grounding in RGBD Images' in CVPR 2021</br>
<br>
Paper - <a href="https://arxiv.org/pdf/2103.07894" target="__blank">ArXiv - pdf</a> (<a href="https://arxiv.org/abs/2103.07894" target="__blank">abs</a>) 
<br>
Project page: https://unclemedm.github.io/Refer-it-in-RGBD/ <br>

### Introduction
We present a novel task of 3D visual grounding in <b>single-view RGB-D images</b> where the referred objects are often only <b>partially scanned</b>. 
In contrast to previous works that directly generate object proposals for grounding in the 3D scenes, we propose a bottom-up approach to gradually aggregate information, effectively addressing the challenge posed by the partial scans. 
Our approach first fuses the language and the visual features at the bottom level to generate a heatmap that coarsely localizes the relevant regions in the RGB-D image. Then our approach adopts an adaptive search based on the heatmap and performs the object-level matching with another visio-linguistic fusion to finally ground the referred object. 
We evaluate the proposed method by comparing to the state-of-the-art methods on both the RGB-D images extracted from the ScanRefer dataset and our newly collected SUN-Refer dataset. Experiments show that our method outperforms the previous methods by a large margin (by 11.1% and 11.2%  Acc@0.5) on both datasets.

### Dataset
<a href="https://unclemedm.github.io/Refer-it-in-RGBD/SUNREFER_v2.json">Download SUNREFER_v2 dataset</a><br>
SUNREFER dataset contains 38,495 referring expression corresponding to 7,699 objects from SUNRGBD dataset. Here is one example from SUNREFER dataset:
<p align="center"><img src="docs/dataset_example.png" width="400px"/></p>

# Install packages
CUDA 10.2 is used for this project. <br>
Install other package by:
```angular2
pip install -r requirement.txt
```
Install weighted FPS by:
```angular2
cd weighted_FPS
python setup.py install
```
Install pointnet2 by:
```angular2
cd third_party/pointnet2
python setup.py install
```
Install MinkowskiEngine, detail can be referred in <a href="https://github.com/NVIDIA/MinkowskiEngine" target="__blank">this link</a>.

# Prepare data
Firstly create a new folder named data under the root directory. Download glove word embedding file glove.p in <a href='http://kaldir.vc.in.tum.de/glove.p' target='__bland'> glove.p</a>.
### ScanRefer dataset
The processed data of ScanRefer and ScanNet is in <a href="https://cuhko365-my.sharepoint.com/:f:/g/personal/115010192_link_cuhk_edu_cn/EpdaZpFCBNBKsV2LxMhf7ckBQiMSv5g6_dBb0bAV2kYRhQ?e=6fP2ri" target="__blank"> processed data</a>.
<br>
 Unzip and put the scannet_singleRGBD folder under data. There should be several folders inside the scannet_singleRGBD,
 which are pcd, storing the point cloud of single-view RGBD image; pose, the camera extrinsic and intrinsic of each image; bbox, store all gt bounding box; and train/val split referring expression data in two .json file.
 

The processing script of how to prepare the data will be released later.

### SUNRefer dataset
The processed data of SUNRefer dataset will be comming in a few days.

# Training
The training procedure is split into two stage.<br>
Firstly, train the voxel-level matching model indenpendently by running
```angular2
python main.py --config ./config/pretrain_config.yaml
```
You can adjust the configuration, I train all the models on one RTX2080Ti using batch size=14.
Then, train the whole referring model by running:
```angular2
python main.py --config ./config/train_scanrefer_config.yaml
```
please make sure the weight of the voxel-level matching is loaded, which is defined in the
`hm_model_resume' entry in the configuration file.
<br>PS: sometime the training will be stopped due to some bugs in CUDA10.x (CUDA11 works fine, but it will need pytorch 1.7.1). You will need to resume the training manually
by setting the resume=True in the configuration file, and change the weight entry to be the path of the checkpoint.
# Testing
Modify the weight path in /config/test_scanrefer_config.yaml. Then run the following command to test the model:
```angular2
python main.py --mode test --config ./config/test_scanrefer_config.yaml
```
