from data.scanrefer_dataloader import singleRGBD_dataset

def get_dataloader(cfg):
    if cfg.data.dataset=="scanrefer-singleRGBD":
        dataset=singleRGBD_dataset(cfg)

    dataloader()