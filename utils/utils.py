from dataset.scanrefer_dataloader import singleRGBD_dataset
from dataset.sunrefer_dataloader import sunrefer_dataset
from torch.utils import data
from network import ref_net
import torch
from loss.voxel_match_loss import voxel_match_loss
from loss.ref_loss import ref_loss
from network.trainer import voxel_match_trainer,ref_trainer
from network.tester import ref_tester
from network.modules.Sparse_UNet import Voxel_Match
from network.ref_net import RGBD_RefNet
import numpy as np
import os
import urllib

class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, **kwargs):
        '''
        initialize model and optimizer.
        :param cfg: configuration file
        :param kwargs: model, optimizer and other specs.
        '''
        self.cfg = cfg
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'min_loss': 1e8})
        self._saved_filename = 'model_last.pth'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        outdict = kwargs
        for k, v in self._module_dict.items():
            if hasattr(v, 'state_dict'):
                outdict[k] = v.state_dict()
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.cfg.config['log']['path'], filename))

    def load(self, filename, *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''

        if self.is_url(filename):
            return self.load_url(filename, *domain)
        else:
            return self.load_file(filename, *domain)

    def parse_checkpoint(self):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config['resume']:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.log_string('Begin to resume from the last checkpoint.')
            self.resume()
        elif self.cfg.config['finetune']:
            # only load net weights.
            self.cfg.log_string('Begin to finetune from the existing weight.')
            self.finetune()
        else:
            self.cfg.log_string('Begin to train from scratch.')

    def finetune(self):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config['weight'], str):
            weight_paths = [self.cfg.config['weight']]
        else:
            weight_paths = self.cfg.config['weight']

        for weight_path in weight_paths:
            if not os.path.exists(weight_path):
                self.cfg.log_string('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, 'net')
                self.cfg.log_string('Weights for finetuning loaded.')

    def resume(self):
        '''
        resume the lastest checkpoint
        :return:
        '''
        checkpoint_root = os.path.dirname(self.cfg.save_path)
        saved_log_paths = os.listdir(checkpoint_root)
        saved_log_paths.sort(reverse=True)

        for last_path in saved_log_paths:
            last_checkpoint = os.path.join(checkpoint_root, last_path, self.saved_filename)
            if not os.path.exists(last_checkpoint):
                continue
            else:
                self.load(last_checkpoint)
                self.cfg.log_string('Last checkpoint resumed.')
                return

        self.cfg.log_string('Warning: resume failed: No checkpoint available. Begin to train from scratch.')

    def load_file(self, filename, *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''

        if os.path.exists(filename):
            self.cfg.log_string('Loading checkpoint from %s.' % (filename))
            checkpoint = torch.load(filename)
            scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.log_string('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''
        for key, value in self._module_dict.items():

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if hasattr(value, 'load_state_dict'):
                    if key != 'voxnet' and key!="refnet":
                        value.load_state_dict(checkpoint[key])
                    else:
                        '''load weights module by module'''
                        value.load_state_dict(checkpoint[key])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.log_string('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in our models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.log_string('Warning: the remaining modules %s in checkpoint are not found in our current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars

def custom_collation_fn(data_batch):
    elem=data_batch[0]
    ret_dict={}
    for key in elem:
        data_list = [d[key] for d in data_batch]
        if key=="vox_feats":
            ret_dict[key]=torch.from_numpy(np.concatenate(data_list,axis=0)).float()
        elif key=="vox_coords":
            ret_dict[key] = [torch.from_numpy(data).float() for data in data_list]
        elif key == 'scene_id' or key == "ann_id" or key == "object_id" or key == "sentence" or key == "image_id":
            ret_dict[key]=data_list
        else:
            ret_dict[key] = torch.tensor(data_list, dtype=torch.float32)

    return ret_dict

def get_dataloader(cfg,mode):
    if cfg['data']['dataset']=="scanrefer-singleRGBD":
        if mode=="train":
            dataset=singleRGBD_dataset(cfg,True)
        elif mode=="test":
            dataset=singleRGBD_dataset(cfg,False)
    elif cfg['data']['dataset']=="sunrefer":
        if mode=="train":
            dataset=sunrefer_dataset(cfg,True)
        elif mode=="test":
            dataset=sunrefer_dataset(cfg,False)

    dataloader=data.DataLoader(dataset=dataset,batch_size=cfg['data']['batch_size'],shuffle=False,num_workers=cfg['data']['num_workers'],drop_last=True,collate_fn=custom_collation_fn)
    return dataloader

def get_optimizer(config, net):
    '''
    get optimizer for networks
    :param config: configuration file
    :param model: nn.Module network
    :return:
    '''
    optim_config=config['optimizer']
    if config['optimizer']['method'] == 'Adam':
        '''collect parameters with specific optimizer spec'''
        if config["method"]=="voxel_match_pretrain":
            optimizer = torch.optim.Adam(net.parameters(),
                                         lr=float(optim_config['lr']),
                                         betas=(optim_config['beta1'],optim_config['beta2']))
            return optimizer
        elif config["method"]=="refer":
            optimizer_hm=torch.optim.Adam(net[0].parameters(),
                                         lr=float(optim_config['lr_hm']),
                                         betas=(optim_config['beta1'],optim_config['beta2']))
            optimizer_ref=torch.optim.Adam(net[1].parameters(),
                                         lr=float(optim_config['lr_ref']),
                                         betas=(optim_config['beta1'],optim_config['beta2']))
            optimizer_list=[optimizer_hm,optimizer_ref]
            return optimizer_list

def load_scheduler(config,optimizer):
    if config["method"]=="voxel_match_pretrain":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[50,80],gamma=config['scheduler']['gamma'])
        return scheduler
    elif config["method"]=="refer":
        scheduler_hm = torch.optim.lr_scheduler.MultiStepLR(optimizer[0], [50, 80], gamma=config['scheduler']['gamma'])
        scheduler_ref= torch.optim.lr_scheduler.MultiStepLR(optimizer[1], [50, 80], gamma=config['scheduler']['gamma'])
        scheduler_list=[scheduler_hm,scheduler_ref]
        return scheduler_list


def get_trainer(config):
    if config["method"] == "voxel_match_pretrain":
        trainer=voxel_match_trainer
    elif config["method"] == "refer":
        trainer=ref_trainer
    return trainer

def get_tester(config):
    if config["method"] == "refer":
        tester = ref_tester
    return tester

def get_loss(config,device):
    if config["method"]=="voxel_match_pretrain":
        loss_func=voxel_match_loss().to(device)
    elif config["method"]=="refer":
        loss_func=ref_loss(config).to(device)
    return loss_func



def get_model(cfg,device):
    if cfg["method"] == "voxel_match_pretrain":
        model=Voxel_Match()
        model=model.to(device)
        return model
    elif cfg["method"] == "refer":
        ref_model=RGBD_RefNet(cfg).to(device)
        hm_model=Voxel_Match()
        print("loadding model from:",cfg["hm_model_resume"])
        checkpoint=torch.load(cfg["hm_model_resume"])
        hm_model.load_state_dict(checkpoint['voxnet'])
        hm_model.to(device)
        model_list=[hm_model,ref_model]
        return model_list

def load_device(cfg):
    '''
    load device settings
    :param config:
    :return:
    '''
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        return torch.device("cuda")
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")

