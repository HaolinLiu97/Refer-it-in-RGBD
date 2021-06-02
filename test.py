from utils.utils import load_device, get_model, get_loss, get_dataloader,CheckpointIO,get_tester

def run(cfg):

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    test_loader = get_dataloader(cfg.config, mode='test')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = get_model(cfg.config, device=device)
    if isinstance(net,list):
        checkpoint.register_modules(voxnet=net[0])
        checkpoint.register_modules(refnet=net[1])
    else:
        checkpoint.register_modules(voxnet=net)

    cfg.log_string('loading loss function')
    loss_func= get_loss(cfg.config,device)


    '''Load trainer'''
    cfg.log_string('Loading tester.')
    tester = get_tester(cfg.config)

    '''Start to train'''
    cfg.log_string('Start to test.')
    #cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    tester(cfg, net, loss_func,test_loader=test_loader,device=device,checkpoint=checkpoint)

    cfg.log_string('Testing finished.')