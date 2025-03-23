import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from pointnet2_ops import pointnet2_utils
import random

var_keys_for_gpu = ['partial', 'gt', 'features', 'o', 'normals',  'normed_gt', 'keypoints', 'partial_keypoints', 'normed_keypoints', 'org_partial', 'partial_r', 'basis_points', 'gt_basis_points']


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x

def make_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()),
                                   'initial_lr': learning_rate}])
    return optimizer

def make_schedular(optimizer, step_size, ratio, last_epoch=-1):
    scheduler = StepLR(optimizer,
                       step_size=step_size,
                       gamma=ratio,
                       last_epoch=last_epoch)
    return scheduler    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def batchify_data(data, split='train'):
    batched_data = {}
    for k, v in data.items():
        if k in var_keys_for_gpu:
            batched_data[k] = var_or_cuda(v)
            if k in ['gt','partial', 'features', 'o', 'normals', 'normed_gt', 'keypoints']:
                data_shape = data[k].shape
                if len(data_shape) == 4:
                    if split == 'train':
                        batched_data[k] = batched_data[k].reshape(data_shape[-3],data_shape[-2], data_shape[-1])
                    else:
                        batched_data[k] = batched_data[k].reshape(data_shape[0],data_shape[-2], data_shape[-1])
                else:
                    if split == 'train':
                        batched_data[k] = batched_data[k].reshape(data_shape[-2], data_shape[-1])
                    else:
                        batched_data[k] = batched_data[k].reshape(data_shape[0], data_shape[-1])
    return batched_data

def prepare_data(data, split='train'):
    batched_data = {}
    for k, v in data.items():
        if k in var_keys_for_gpu:
            batched_data[k] = var_or_cuda(v)
    return batched_data


def test_single_epoch(model, dataloader, loss_object):
    model.eval()
    epoch_losses = []
    multiplier = 1e3
        
    for (taxonomy_ids, model_ids, data) in dataloader:

        fixed_data = batchify_data(data, split='val')
        model_outs = model(fixed_data['partial'], fixed_data['features'], fixed_data['o'], fixed_data['normals'])
        point_preds, distance_preds, seeds = model_outs
        loss_val, loss_list = loss_object.get_val_loss(distance_preds, seeds, fixed_data['keypoints'], fixed_data['gt'])
        
        losses = [ls*multiplier for ls in loss_list]
        loss_vals = [l.detach().cpu().numpy().tolist() for l in losses]
        
        epoch_losses.append(loss_vals)

    return np.mean(epoch_losses, axis=0)

def train_single_epoch(model, dataloader, optim, loss_object=None, accum_iter=1, use_progress=False, add_consistency=True):
    model.train()
    epoch_losses = []
    multiplier = 1e3
    iter_num = 0
    
    if use_progress:
        pbar = tqdm(total=len(dataloader))
        
    for (taxonomy_ids, model_ids, data) in dataloader:

        fixed_data = batchify_data(data)
        model_outs = model(fixed_data['partial'], fixed_data['features'], fixed_data['o'], fixed_data['normals'])
        point_preds, distance_preds, seeds = model_outs
        loss_val, loss_list = loss_object.get_loss(distance_preds, seeds, fixed_data['keypoints'], fixed_data['gt'], add_consistency=add_consistency)
        
        loss_val = loss_val / accum_iter
        loss_val.backward()
        
        if((iter_num + 1) % accum_iter == 0) or (iter_num + 1 == len(dataloader)):    
            optim.step()
            optim.zero_grad()
                
        losses = [ls*multiplier for ls in loss_list]
        loss_vals = [l.detach().cpu().numpy().tolist() for l in losses]
        
        epoch_losses.append(loss_vals)

        # Be sure there are some data to show
        if use_progress:
            mean_epoch_loss = np.mean(epoch_losses, axis=0)
            pbar.set_description(
                    '[Batch: %d/%d]' % (iter_num, len(dataloader)))
            pbar.set_postfix(
                loss='%s' % ['%.2f' % l for l in mean_epoch_loss],
            )
            
            pbar.update(1)        
        iter_num += 1
    
    if use_progress:
        pbar.close()
    
    return np.mean(epoch_losses, axis=0)

def train_model(model, dataloader, optim, schedular, loss_object, epochs, accum_iter=1, use_batch_progress=False, add_consistency=True, model_save_dir=""):
    best_losses = [2000, 2000, 2000, 2000, 2000, 2000]
    e = 0
    pbar = tqdm(total=epochs)
    training_losses = []
    learning_rates = []
    
    for e in range(epochs):
        epoch_loss = train_single_epoch(model, dataloader, optim, loss_object, accum_iter=accum_iter, use_progress=use_batch_progress, add_consistency=add_consistency)

        
        if (epoch_loss[0] < best_losses[0]) and (epoch_loss[1] < best_losses[1]):
            best_losses[0] = epoch_loss[0]
            best_losses[1] = epoch_loss[1]
        if model_save_dir != "":
            save_model(model, model_save_dir)
        
        training_losses.append(epoch_loss)
        schedular.step()
        pbar.set_description(
                '[Epoch %d/%d]' % (e, epochs))
        pbar.set_postfix(
            b_loss='%s' % ['%.2f' % l for l in best_losses],
            c_loss='%s' % ['%.2f' % l for l in epoch_loss]
        )
        pbar.update(1)
        e += 1
        learning_rates.append(get_lr(optim))
    pbar.close()
    return best_losses, np.stack(training_losses, axis=0), learning_rates

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()