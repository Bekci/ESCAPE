import os
import torch
import numpy as np
from datasets.utils import read_yaml
from datasets.PCN import get_dataset_and_loader
from models.pointr.escape import ESCAPE
from tqdm import tqdm
from utils.training import var_or_cuda, var_keys_for_gpu
from loss_functions.chamfer_ndim import CDNLoss
from utils.optimization import find_points_from_distance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n_k', '--num_keypoint', type=int, default=8)
parser.add_argument('-cp', '--config_path', type=str, default='configs/noise_exps/denoise128.yaml')
parser.add_argument('-mp', '--model_path', type=str, default='')
parser.add_argument('-k', '--keypoint', type=str, default='curvature_region')
parser.add_argument('-c_k', '--curve_k', type=int, default=16)
parser.add_argument('-c_r', '--curve_radius', type=float, default=0.075)
parser.add_argument('-c_t', '--curve_thres', type=float, default=0.5)


def test_single_epoch(model, dataloader, loss_object):
    model.eval()
    epoch_losses = []
    multiplier = 1e3
    
    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in tqdm(dataloader):

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


def optimize_points_loader(model, dataloader, loss_object, validate_loader=None):
    losses = []
    sup_losses = []
    model.eval()
    results_per_cat = {}
    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in tqdm(dataloader):

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])

            for i in range(ret[0].shape[0]):
        
                coarse_points = ret[0][i]
                dense_points = ret[-1][i]
                category_id = taxonomy_ids[i]
                
                basis_points_np = data['basis_points'].cpu().numpy()[i]
                distances_np  = dense_points.cpu().numpy()
                gt_dists_np = gt_dists.cpu().numpy()[i]
                
                optimized_points = find_points_from_distance(distances_np, basis_points_np)
                loss = loss_object.chamfer_loss(torch.unsqueeze(torch.from_numpy(optimized_points.astype('float32')), 0).cuda(), data['gt'][i:i+1])
                loss_val = loss.item()
                losses.append(loss_val)

                if category_id not in results_per_cat:
                    results_per_cat[category_id] = []
                
                results_per_cat[category_id].append(loss_val * 1000)
            
            if validate_loader:
                test_single_epoch(model, validate_loader, loss_object)

    return losses, results_per_cat


def optimize_points_loader(model, dataloader, loss_object, validate_loader=None):
    losses = []
    sup_losses = []
    model.eval()

    results_per_cat = {}
    with torch.no_grad():
        for (taxonomy_ids, model_ids, data) in tqdm(dataloader):

            for k in data:
                if k in var_keys_for_gpu:
                    data[k] = var_or_cuda(data[k])

            gt_dists = torch.sqrt(torch.sum(1e-6 +  (data['gt'][:,:,None,:] - data['basis_points'][:,None,:,:])**2, axis=-1))

            ret = model(data['partial'], data['basis_points'])

            for i in range(ret[0].shape[0]):
        
                coarse_points = ret[0][i]
                dense_points = ret[-1][i]
                category_id = taxonomy_ids[i]
                
                basis_points_np = data['basis_points'].cpu().numpy()[i]
                distances_np  = dense_points.cpu().numpy()
                gt_dists_np = gt_dists.cpu().numpy()[i]
                gt_points_np = data['gt'][i].cpu().numpy()
                  
                optimized_points = find_points_from_distance(distances_np, basis_points_np)
                loss = loss_object.chamfer_loss(torch.unsqueeze(torch.from_numpy(optimized_points.astype('float32')), 0).cuda(), data['gt'][i:i+1])
                loss_val = loss.item()
                losses.append(loss_val)

                if category_id not in results_per_cat:
                    results_per_cat[category_id] = []
                
                results_per_cat[category_id].append(loss_val * 1000)
            
            if validate_loader:
                test_single_epoch(model, validate_loader, loss_object)

    return losses, results_per_cat



# ['02691156', '02933112', '02958343',  '03001627',  '03636649',  '04256520',  '04379243', '04530566']
# ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'watercraft']


def get_curvature_exp_name(curv_dict):
    return "k_{}_cr_{}_th_{}".format(
        curv_dict['k'],
        curv_dict['curvature_radius'],
        curv_dict['curvature_thres']
    )
if __name__ == '__main__':
    
    args = parser.parse_args()
    keypoint_number = args.num_keypoint
    keypoint_type = args.keypoint
    model_config_path = args.config_path
    model_save_path = args.model_path
    curvature_params = {}
    curvature_params['k'] =  args.curve_k
    curvature_params['curvature_radius'] = args.curve_radius
    curvature_params['curvature_thres'] = args.curve_thres

    print("Keypoint Type: ", keypoint_type ,  " Model config: ", curvature_params)

    config_path = '/mnt/projects/PointCloudCompletion/code/BEST/configs/roi_pcn_skeleton.yaml'
    config = read_yaml(config_path)

    model_config = read_yaml(model_config_path)

    model = ESCAPE(model_config.model).cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    saved_model_config = torch.load(model_save_path)

    model.load_state_dict(saved_model_config['model_state_best_val'])
    
    print("VALIDATION")
    
    sample_num = 50000
    val_dataset, val_loader = get_dataset_and_loader('val', config, 1, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)

    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1') 

    val_score = test_single_epoch(model, val_loader, loss_obj)
    print(" val loss: ", val_score)
    
    opt_val_dataset, opt_val_loader = get_dataset_and_loader('val', config, 4, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
    
    val_dataset, val_loader = get_dataset_and_loader('val', config, 8, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
    
    point_losses, opt_res_per_cat = optimize_points_loader(model, opt_val_loader, loss_obj, validate_loader=val_loader)
    
    mean_loss = np.mean(point_losses)
    print("Optimized point loss: %.2f " % (mean_loss*1e3))
    
    for cat in opt_res_per_cat:
        print("Distance loss %.4f for %s" % (np.mean(opt_res_per_cat[cat]), cat))

    print("TEST")
    
    sample_num = 50000
    val_dataset, val_loader = get_dataset_and_loader('test', config, 1, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)

    loss_obj = CDNLoss(mode="point", metric_fun='cd_l1') 

    val_score = test_single_epoch(model, val_loader, loss_obj)
    print(" test loss: ", val_score)
    
    opt_val_dataset, opt_val_loader = get_dataset_and_loader('test', config, 4, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
    
    val_dataset, val_loader = get_dataset_and_loader('val', config, 8, sample_num, shuffle=False, drop_last=False, num_keypoints=keypoint_number, keypoint_type=keypoint_type, curvature_params=curvature_params)
    
    point_losses, opt_res_per_cat = optimize_points_loader(model, opt_val_loader, loss_obj, validate_loader=None)
    
    mean_loss = np.mean(point_losses)
    print("Optimized point loss: %.2f " % (mean_loss*1e3))
    
    for cat in opt_res_per_cat:
        print("Distance loss %.4f for %s" % (np.mean(opt_res_per_cat[cat]), cat))