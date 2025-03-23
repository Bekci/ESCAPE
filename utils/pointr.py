import torch
import torch.optim as optim
from models.pointr.misc import build_lambda_bnsche, build_lambda_sche, GradualWarmupScheduler
from timm.scheduler import CosineLRScheduler

def build_optimizer(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, base_model.parameters()), **opti_config.kwargs)
    else:
        raise NotImplementedError()

    return optimizer

def build_scheduler(base_model, optimizer, config, last_epoch=-1):
    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs, last_epoch=last_epoch)  # misc.py
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs)
    elif sche_config.type == 'GradualWarmup':
        scheduler_steplr = torch.optim.lr_scheduler.StepLR(optimizer, last_epoch=last_epoch, **sche_config.kwargs_1)
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_steplr, **sche_config.kwargs_2)
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.t_max,
                lr_min=sche_config.kwargs.min_lr,
                warmup_t=sche_config.kwargs.initial_epochs,
                t_in_epochs=True)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return scheduler