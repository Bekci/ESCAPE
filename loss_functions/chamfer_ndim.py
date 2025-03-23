import torch
from chamfer_distance import ChamferDistance as chamfer_dist
from pointnet2_ops.pointnet2_utils import gather_operation, furthest_point_sample

def fps_subsample(pcd, n_points=2048):
    """
    Args
        pcd: (b, 16384, 3)

    returns
        new_pcd: (b, n_points, 3)
    """
    new_pcd = gather_operation(pcd.permute(0, 2, 1).contiguous(), furthest_point_sample(pcd, n_points))
    new_pcd = new_pcd.permute(0, 2, 1).contiguous()
    return new_pcd

class CDNLoss:
    def __init__(self, mode='both', metric_fun='cd_l1'):        
        if metric_fun == 'cd_l1':
            self.chd = chamfer_dist()
            self.metric = self.chamfer_loss
        elif metric_fun == 'l1':
            self.metric = torch.nn.L1Loss()
        elif metric_fun == 'l2':
            self.metric = torch.nn.MSELoss()
        else:
            raise ValueError("wrong loss function name: %s" % metric_fun)

        
        self.training_mode = mode

    def chamfer_loss(self, prediction, gt_points):
        d1, d2, _, _ = self.chd(prediction, gt_points)
        d1 = torch.clamp(d1, min=1e-9)
        d2 = torch.clamp(d2, min=1e-9)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2
        
    
    def calculate_loss_single_dim(self, prediction, basis, gt_points, radius=None):
        eps = 1e-6
        b, n_points, n_basis = prediction.shape
        gt_dists = torch.sqrt(torch.sum(eps +  (gt_points[:,:,None,:] - basis[:,None,:,:])**2, axis=-1))
        
        if radius is not None:
            gt_dists = gt_dists / radius.view(-1,1,1)
            prediction = prediction / radius.view(-1,1,1)

        loss = self.metric(prediction, gt_dists)
        return loss
        
    def get_distance_loss(self, dist_preds, keypoints, gt_points, radius=None):
        dc, d1, d2, d3 = dist_preds
    
        gt_d2 = fps_subsample(gt_points, d2.shape[1])
        gt_d1 = fps_subsample(gt_d2, d1.shape[1])
        gt_dc = fps_subsample(gt_d1, dc.shape[1])
    
        lossc = self.calculate_loss_single_dim(dc, keypoints, gt_dc, radius)
        loss1 = self.calculate_loss_single_dim(d1, keypoints, gt_d1, radius)
        loss2 = self.calculate_loss_single_dim(d2, keypoints, gt_d2, radius)
        loss3 = self.calculate_loss_single_dim(d3, keypoints, gt_points, radius)
    
        loss_all = lossc + loss1 + loss2 + loss3
        losses = [lossc, loss1, loss2, loss3]
        
        return loss_all, losses   

    def get_distance_loss_single(self, dist_preds, keypoints, gt_points):
        cur_level_gt = gt_points
        
        loss_all = 0.0
        losses = []

        for dist in dist_preds[::-1]:
            
            cur_level_gt = fps_subsample(cur_level_gt, dist.shape[1])
            cur_loss = self.calculate_loss_single_dim(dist, keypoints, cur_level_gt)
            loss_all += cur_loss
            losses = losses + [cur_loss]
    
        return loss_all, losses           
    

    def get_loss(self, distance_preds, seed_preds, gt_keypoints, gt_points, radius=None):
        
        if self.training_mode == 'point':
            return self.get_org_point_loss(distance_preds, gt_points)
        
        if self.training_mode == 'distance' or self.training_mode == 'ppf':
            return self.get_distance_loss(distance_preds, seed_preds, gt_points)
        
        if self.training_mode == 'keypoint':
            seed_loss = self.metric(seed_preds, gt_keypoints)
            return seed_loss, [seed_loss]
    
        return None, None
    
    def get_org_point_loss(self, point_preds, gt_points):
        pc, p1, p2, p3 = point_preds
    
        gt_d2 = fps_subsample(gt_points, p2.shape[1])
        gt_d1 = fps_subsample(gt_d2, p1.shape[1])
        gt_dc = fps_subsample(gt_d1, pc.shape[1])
    
        lossc = self.chamfer_loss(pc, gt_dc)
        loss1 = self.chamfer_loss(p1, gt_d1)
        loss2 = self.chamfer_loss(p2, gt_d2)
        loss3 = self.chamfer_loss(p3, gt_points)
    
        loss_all = lossc + loss1 + loss2 + loss3
        losses = [lossc, loss1, loss2, loss3]
        
        return loss_all, losses
    