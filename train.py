from utils.utils import get_optimizer, load_scheduler, load_device, get_model, get_loss, get_dataloader,get_trainer,CheckpointIO

def run(cfg):

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))
    checkpoint=CheckpointIO(cfg)

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    train_loader = get_dataloader(cfg.config, mode='train')
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

    '''Load optimizer'''
    cfg.log_string('Loading optimizer.')
    optimizer = get_optimizer(config=cfg.config, net=net)
    if isinstance(net, list):
        checkpoint.register_modules(voxopt=optimizer[0])
        checkpoint.register_modules(refopt=optimizer[1])
    else:
        checkpoint.register_modules(voxopt=optimizer)

    '''Load scheduler'''
    cfg.log_string('Loading optimizer scheduler.')
    scheduler = load_scheduler(config=cfg.config, optimizer=optimizer)
    if isinstance(net, list):
        checkpoint.register_modules(voxsch=scheduler[0])
        checkpoint.register_modules(refsch=scheduler[1])
    else:
        checkpoint.register_modules(voxsch=scheduler)

    '''Load trainer'''
    cfg.log_string('Loading trainer.')
    trainer = get_trainer(cfg.config)

    '''Start to train'''
    cfg.log_string('Start to train.')
    #cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    trainer(cfg, net, loss_func, optimizer,scheduler,train_loader=train_loader, test_loader=test_loader,device=device,checkpoint=checkpoint)

    cfg.log_string('Training finished.')