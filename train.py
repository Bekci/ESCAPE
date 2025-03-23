import os
import torch
import numpy as np
from datasets.utils import read_yaml
from datasets.PCN import get_dataset_and_loader
from models.pointr.escape import ESCAPE
from tqdm import tqdm
from utils.training import var_or_cuda, var_keys_for_gpu, save_model
from loss_functions.chamfer_ndim import CDNLoss
from utils.visualization import save_loss_graphs
from utils.pointr import build_optimizer, build_scheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n_k', '--num_keypoint', type=int, default=8)
parser.add_argument('-cp', '--config_path', type=str, default='configs/escape.yaml')
parser.add_argument('-tcp', '--train_config_path', type=str, default='configs/pcn.yaml')
parser.add_argument('-k', '--keypoint', type=str, default='curvature_radius')
parser.add_argument('-c_k', '--curve_k', type=int, default=16)
parser.add_argument('-c_r', '--curve_radius', type=float, default=0.075)
parser.add_argument('-c_t', '--curve_thres', type=float, default=0.5)
parser.add_argument('-c_n', '--curvature_neighbor', type=int, default=16)
parser.add_argument('-sr', '--sample_ratio', type=float, default=1.0)


def test_single_epoch(model, dataloader, loss_object):
    model.eval()
    epoch_losses = []
    multiplier = 1e3
    
    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in dataloader:

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])

            coarse_points = ret[0]
            dense_points = ret[-1]

            sparse_loss = loss_object.chamfer_loss(coarse_points, gt_dists)
            dense_loss = loss_object.chamfer_loss(dense_points, gt_dists)
            loss_vals = [sparse_loss.item() * 1000, dense_loss.item() * 1000]
            
            epoch_losses.append(loss_vals)

    return np.mean(epoch_losses, axis=0)

def train_single_epoch(model, dataloader, optim, current_epoch, loss_object=None, accum_iter=1, use_progress=False):
    model.train()
    epoch_losses = []
    multiplier = 1e3
    iter_num = 0
    if use_progress:
        pbar = tqdm(total=len(dataloader))
        
    for (taxonomy_ids, model_ids, data) in dataloader:

        for k in data:
            if k in var_keys_for_gpu:
                data[k] = var_or_cuda(data[k])
        
        gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))
        ret = model(data['partial'], data['basis_points'])
        
        sparse_loss, dense_loss = model.module.get_loss(ret, gt_dists)

        _loss = sparse_loss + dense_loss 

        _loss = _loss / accum_iter
        _loss.backward()
        
        if((iter_num + 1) % accum_iter == 0) or (iter_num + 1 == len(dataloader)):
            optim.step()
            model.zero_grad()
        
        loss_vals = [sparse_loss.item() * 1000, dense_loss.item() * 1000]
        epoch_losses.append(loss_vals)

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

def train_model(model, dataloader, optim, schedular, loss_object, epochs, val_dataloader=None, accum_iter=1, use_batch_progress=False, model_save_dir=""):
    best_losses = [2000, 2000, 2000, 2000]
    best_val_losses = [2000, 2000, 2000, 2000]
    e = 0
    
    training_losses = []
    val_losses = []
    
    best_train_model = os.path.join(model_save_dir, "train_best.pth")
    best_val_model = os.path.join(model_save_dir, "val_best.pth")
    cur_train_config = os.path.join(model_save_dir, "train_last.pth")
    
    val_loss_last_improve = 0
    patience = 50

    if os.path.exists(cur_train_config):
        print("Found final training, continue training...")
        final_config = torch.load(cur_train_config)
        initial_epoch = final_config['epoch']
        best_val_losses = final_config['best_val_loss']
        best_losses = final_config['best_loss']
        optim.load_state_dict(final_config['optimizer_state'])
        schedular.load_state_dict(final_config['scheduler_state'])
        model.load_state_dict(final_config['model_state'])
    else:
        initial_epoch = 0
    
    model.zero_grad()
    pbar = tqdm(total=epochs)
    for e in range(initial_epoch, initial_epoch+epochs):
        
        epoch_loss = train_single_epoch(model, dataloader, optim, e, loss_object, accum_iter=accum_iter, use_progress=use_batch_progress)
        training_losses.append(epoch_loss)
        
        if epoch_loss[-1] < best_losses[-1]:
            best_losses[-1] = epoch_loss[-1]
            if model_save_dir != "":
                save_model(model, best_train_model)

        if val_dataloader:
            val_loss = test_single_epoch(model, val_dataloader, loss_object)
            val_losses.append(val_loss)

            if val_loss[-1] < best_val_losses[-1]:
                
                best_val_losses[-1] = val_loss[-1]

                print("Epoch: %d Best val loss improved to: %.2f" % (e, val_loss[-1]) )
                val_loss_last_improve = e

                if model_save_dir != "":
                    torch.save(
                        {
                        'epoch': e,                
                        'model_state_best_val': model.state_dict(),
                        'best_loss': best_losses,
                        'current_loss': epoch_loss,
                        'best_val_loss': best_val_losses
                        }, best_val_model)                    
                
        if isinstance(schedular, list):
            for item in schedular:
                item.step()
        else:
            schedular.step()

        pbar.set_description(
                '[Epoch %d/%d]' % (e, initial_epoch+epochs))
        pbar.set_postfix(
            b_loss='%s' % ['%.2f' % l for l in best_losses],
            c_loss='%s' % ['%.2f' % l for l in epoch_loss]
        )
        pbar.update(1)
        e += 1

        if model_save_dir != "":
            torch.save(
                {
                'epoch': e,                
                'model_state': model.state_dict(),
                'optimizer_state': optim.state_dict(),
                'best_loss': best_losses,
                'current_loss': epoch_loss,
                'best_val_loss': best_val_losses
                }, cur_train_config)

        if (e - val_loss_last_improve) > patience:
            print("Val loss not improved between {}-{}. Early stopping...".format(e, val_loss_last_improve))
            break
    pbar.close()
    return best_losses, np.stack(training_losses, axis=0), np.stack(val_losses, axis=0)

def get_curvature_exp_name(curv_dict):
    return "k_{}_cr_{}_th_{}_cn_{}".format(
        curv_dict['k'],
        curv_dict['curvature_radius'],
        curv_dict['curvature_thres'],
        curv_dict['curvature_neighbor']
    )


if __name__ == '__main__':
    
    args = parser.parse_args()
    keypoint_number = args.num_keypoint
    keypoint_type = args.keypoint
    model_config_path = args.config_path
    dataset_ratio = args.sample_ratio
    train_config_path = args.train_config_path
    
    curvature_params = {}
    curvature_params['k'] =  args.curve_k
    curvature_params['curvature_radius'] = args.curve_radius
    curvature_params['curvature_thres'] = args.curve_thres
    curvature_params['curvature_neighbor'] = args.curvature_neighbor
    

    config = read_yaml(train_config_path)

    model_config = read_yaml(model_config_path)

    model = ESCAPE(model_config.model).cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    exp_base_path = os.path.join('train_res', keypoint_type, get_curvature_exp_name(curvature_params),  "k_{}".format(keypoint_number))
    
    os.makedirs(exp_base_path, exist_ok=True)
    
    tr_cache_dir = os.path.join('cache', keypoint_type, "k{}".format(keypoint_number), get_curvature_exp_name(curvature_params), "train")
    os.makedirs(tr_cache_dir, exist_ok=True)

    sample_num = 40000
    num_epochs = 250
    step_size = 40

    tr_dataset, tr_loader = get_dataset_and_loader('train', config, 16, sample_num, shuffle=True, drop_last=True, cache_dir=tr_cache_dir, 
                                                   num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params,
                                                   dataset_size=dataset_ratio)
    val_dataset, val_loader = get_dataset_and_loader('val', config, 1, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)

    optim = build_optimizer(model, model_config)
    scheduler = build_scheduler(model, optim, model_config)

    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1') 
    
    models_save_dir = os.path.join(exp_base_path, "models")
    os.makedirs(models_save_dir, exist_ok=True)

    best_loss, training_losses, validation_losses = train_model(model, tr_loader, 
                                                            optim, scheduler, loss_obj, num_epochs,
                                                            val_dataloader=val_loader, use_batch_progress=True, 
                                                            model_save_dir=models_save_dir)

    print("Best loss: ", best_loss)

    save_loss_graphs(training_losses, exp_base_path, "train")
    save_loss_graphs(validation_losses, exp_base_path, "val")

    val_score = test_single_epoch(model, val_loader, loss_obj)
    print(" val loss: ", val_score)
