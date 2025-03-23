import os
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.data_transforms import Compose, deterministic_fps, fps_with_initial_samples
from datasets.utils import read_pcd, to_o3d_pcd
import random
import json
from datasets.curvature import select_points_by_curvature, compute_curvature_normal, get_basis_improved_curv_points, get_improved_curv_points_from_initial, get_point_with_max_curv_around, compute_curvature_descriptor


def get_dataset_and_loader(split, config, batch_size, n_instance, shuffle=True, drop_last=True, transform_json_path="", apply_deform_on_gt=True, cache_dir="", num_keypoints=8, keypoint_type='curvature_radius', curvature_params={}, dataset_size=1.0, noise_ratio=0.0, drop_ratio=0.0):
    pcn_dataset = BatchedPCN(split, config, n_instance, rotation_json=transform_json_path, deform_gt=apply_deform_on_gt, cache_dir=cache_dir, n_keypoints=num_keypoints, load_car_only=False, 
                             keypoint_type=keypoint_type, curvature_params=curvature_params, sampling_ratio=dataset_size, noise_ratio=noise_ratio, drop_ratio=drop_ratio)
    pcn_dataloader = DataLoader(pcn_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return pcn_dataset, pcn_dataloader

def get_dataset_and_loader_for_car(split, config, batch_size, n_instance, shuffle=True, drop_last=True, transform_json_path="", apply_deform_on_gt=True, cache_dir="", num_keypoints=8, keypoint_type='curvature_radius', curvature_params={}):
    pcn_dataset = BatchedPCN(split, config, n_instance, rotation_json=transform_json_path, deform_gt=apply_deform_on_gt, cache_dir=cache_dir, n_keypoints=num_keypoints, load_car_only=True, keypoint_type=keypoint_type, curvature_params=curvature_params)
    pcn_dataloader = DataLoader(pcn_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return pcn_dataset, pcn_dataloader


def construct_inputs(point_tensor):
    np_points = point_tensor.numpy()
    o3d_src_pcd = to_o3d_pcd(np_points)
    
    o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
    o3d_src_pcd.orient_normals_to_align_with_direction()
    src_normals = np.asarray(o3d_src_pcd.normals)
    #src_normals = normal_redirect(np_points, src_normals, view_point=[0.,0.,0.])

    src_o = np.array([np_points.shape[0]], dtype=np.int32)
    src_feats = np.ones(shape=(np_points.shape[0], 1))
    return np_points.astype(np.float32), src_feats.astype(np.float32), src_o, src_normals.astype(np.float32)

def calculate_keypoint_normals(keypoints):

    keypoint_pcd = to_o3d_pcd(keypoints)
    
    keypoint_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
    
    keypoint_pcd.orient_normals_to_align_with_direction()
    src_normals = np.asarray(keypoint_pcd.normals)

    return src_normals.astype(np.float32)


class BatchedPCN(Dataset):
    def __init__(self, split, config, n_instance, rotation_json="", deform_gt=True, cache_dir="", 
                 n_keypoints=8, load_car_only=False, keypoint_type='basis', curvature_params={}, sampling_ratio=1.0, noise_ratio=0.0, drop_ratio=0.0):
        
        self.partial_points_parent_path = config.dataset.partial_points_parent_path % split
        self.complete_points_parent_path = config.dataset.complete_points_parent_path % split
        self.only_car = load_car_only
        self.npoints = config.dataset.n_points
        self.subset = split
        self.noise = noise_ratio
        self.deform_gt = deform_gt
        self.n_instance = n_instance
        self.n_keypoints = n_keypoints
        self.curvature_params = curvature_params
        self.keypoint_type = keypoint_type
        self.applied_rotation = {}
        self.drop_percent = drop_ratio
        self.cache_dir = cache_dir
        self.gt_point_cloud_cache_dir = './cache/gt_pcn'
        self.subset_ratio = sampling_ratio
        
        if self.cache_dir != "":
            os.makedirs(self.cache_dir, exist_ok=True)

        self.transforms = self._get_transforms(self.subset)
        self.render_idx = [0, 1, 2, 3, 4, 5, 6, 7] if self.subset == 'train' else [0]
        
        self.n_renderings = len(self.render_idx)

        self.file_list = self._get_file_list(self.subset)
        
        self.transformations = None
        

        if rotation_json != "":
            with open(rotation_json, 'rb') as jfile:
                self.transformations = json.load(jfile)

        self.gt_cache = {}
        self.partial_cache = {}
        

    def _get_transforms(self, subset):
        if subset == 'train':
            return Compose([{
                'callback': 'SelectPoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['partial']
            },
            {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }
            ])
        else:
            return Compose([{
                'callback': 'SelectPoints',
                'parameters': {
                    'n_points': self.npoints
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []
        sub_folder_name = 'gt' if subset == 'train' else 'gt_val'
        categories = os.listdir(self.partial_points_parent_path)
        
        if self.only_car:
            categories = ['02958343']
        
        for dc in categories:
            category_partial_path = os.path.join(self.partial_points_parent_path, dc)
            category_complete_path = os.path.join(self.complete_points_parent_path, dc)
            samples = os.listdir(category_partial_path)
            samples.sort()

            for s in samples:
                file_list.append({
                    'taxonomy_id': dc,
                    'model_id': s,
                    'partial_path': os.path.join(category_partial_path, s),
                    'gt_path':os.path.join(category_complete_path, "%s.pcd" % s),
                    })
        
        np.random.shuffle(file_list)
        num_samples = int(len(file_list) * self.subset_ratio)
        selected_files = np.random.choice(file_list, size=num_samples, replace=False)
        return selected_files


    def __getitem__(self, idx):
        
        # Get the first index if it is in sanity check mode
        if idx > (len(self.file_list) - 1):
            sample = self.file_list[0].copy()

        else:
            sample = self.file_list[idx].copy()

        instance_category = sample['taxonomy_id']
        instance_sample =  sample['model_id']
        rand_ind = random.randint(0, self.n_renderings - 1) if self.subset=='train' else 0
        view_idx = self.render_idx[rand_ind]

        data = self._load_gt(instance_category, instance_sample, sample['gt_path'])
        
        cur_partial_path = os.path.join(sample['partial_path'], "%02d.pcd" % view_idx)
        data_partial = self._load_sample(instance_category, instance_sample, view_idx, cur_partial_path)

        data['basis_points'] = data_partial['basis_points'].copy()
        data['partial'] = data_partial['partial'].clone()
        data['n_initial_points'] = data_partial['n_initial_points']

        return sample['taxonomy_id'], sample['model_id'], data

    
    def _load_gt(self, category_name, sample_name, file_path):
        cache_key = "{}_{}_gt.npy".format(category_name, sample_name)
        
        cached_data = self._check_gt_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        data = {}
        if self.deform_gt:
            data['gt'] = read_pcd(file_path).astype(np.float32)
        else:
            data['gt'] = read_pcd(file_path, deformation=None).astype(np.float32)
        
        if self.transforms is not None:
            data = self.transforms(data)
            
        if self.subset == "train":
            self._write_gt_cache(cache_key, data)

        return data
    
    def _check_gt_cache(self, cache_key):
        try:
            # Check file cache
            if self.gt_point_cloud_cache_dir != "":
                cache_file_path = os.path.join(self.gt_point_cloud_cache_dir, "gt", cache_key)
                if os.path.exists(cache_file_path):
                    with open(cache_file_path, 'rb') as npfile:
                        load_data = np.load(npfile, allow_pickle=True)
                        return load_data.item()
        except:
            return None
        return None

    def _write_gt_cache(self, cache_key, data):
        try:
            # Check file cache
            if self.gt_point_cloud_cache_dir != "":
                cache_gt_main_dir = os.path.join(self.gt_point_cloud_cache_dir, "gt")
                os.makedirs(cache_gt_main_dir, exist_ok=True)
                cache_file_path = os.path.join(cache_gt_main_dir, cache_key)
                with open(cache_file_path, 'wb') as npfile:
                    np.save(npfile, data)
        except:
            return None
        return None


    def _load_sample(self, category_name, sample_name, index, file_path):
        cache_key = "{}_{}_{}.npy".format(category_name, sample_name, index)

        cached_data = self._check_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        data = {}

        partial_pcd = o3d.io.read_point_cloud(file_path)
        if self.drop_percent > 0.0:
            n_points = np.array(partial_pcd.points).shape[0]
            n_points_to_drop = int(self.drop_percent * n_points)
            partial_pcd = partial_pcd.farthest_point_down_sample(n_points-n_points_to_drop)
    
        partial_points = np.array(partial_pcd.points)
        initial_point_num = partial_points.shape[0]
        partial_points = partial_points + np.random.normal(size=partial_points.shape) * self.noise
        partial_pcd = to_o3d_pcd(partial_points)
        
        if self.keypoint_type == "curvature":
            curvature_points = self._get_curvature_points(partial_pcd)

        elif self.keypoint_type == "curvature_region":
            curvature_points = self._get_curvature_points_from_region(partial_pcd,
                                                                      self.n_keypoints,
                                                                      k=self.curvature_params['k'],
                                                                      thres=self.curvature_params['curvature_thres']
                                                                      )
        elif self.keypoint_type == "curvature_radius":
            curvature_points = self._get_curvature_points_from_radius(partial_pcd,
                                                                      partial_points,
                                                                      self.n_keypoints,
                                                                      k=self.curvature_params['k'],
                                                                      radius=self.curvature_params['curvature_radius']
                                                                      )
        elif self.keypoint_type == "basis_improved":
            curvature_points = get_basis_improved_curv_points(partial_pcd, 
                                                              partial_points,
                                                              self.curvature_params['k'],
                                                              self.n_keypoints,
                                                              self.curvature_params['neigbor_size'])
        elif self.keypoint_type == "skeleton_improved":
            skeleton_points = self._get_keypoints(category_name, sample_name, index)
            curvature_points = get_improved_curv_points_from_initial(partial_pcd, 
                                                              partial_points,
                                                              skeleton_points,
                                                              self.curvature_params['k'],
                                                              self.curvature_params['neigbor_size'])
        elif self.keypoint_type == "mean_curv":
            curvature_points = get_point_with_max_curv_around(partial_pcd, 
                                                              partial_points,
                                                              self.curvature_params['k'],
                                                              self.n_keypoints,
                                                              self.curvature_params['neigbor_size'])

        elif self.keypoint_type == "snake_based_curv":
            curvature_points = self._get_desc_curvature_points_from_radius(
                category_name,
                sample_name,
                index,
                partial_pcd, 
                partial_points,
                self.n_keypoints,
                k=self.curvature_params['k'],
                radius=self.curvature_params['curvature_radius'])

        elif self.keypoint_type == "snake" or self.keypoint_type == "skeleton":
            curvature_points = self._get_keypoints(category_name, sample_name, index)

        data['partial'] = partial_points.astype(np.float32)
        
        if self.transforms is not None:
            data = self.transforms(data)

        if self.keypoint_type == "basis":
            basis_points = deterministic_fps(data['partial'].numpy(), self.n_keypoints).astype(np.float32)
            data['basis_points'] = basis_points
        
        elif self.keypoint_type in ["curvature", "curvature_region", "snake", "skeleton", "curvature_radius", "basis_improved", "skeleton_improved", "mean_curv", "snake_based_curv"]:
            data['basis_points'] = curvature_points.astype(np.float32)
        
        if self.subset == "train":
            self._write_to_cache(cache_key, data, "partial")

        data['n_initial_points'] = initial_point_num
        return data

    def _check_cache(self, cache_key):
        try:
            # Check file cache
            if self.cache_dir != "":
                item_cache_dir = os.path.join(self.cache_dir, cache_key)
                if os.path.exists(item_cache_dir):
                    with open(item_cache_dir, 'rb') as npfile:
                        load_data = np.load(npfile, allow_pickle=True)
                        return load_data.item()
        except:
            return None
                    
        # Check memory cache
        if cache_key in self.gt_cache:
            return self.gt_cache[cache_key]
        
        elif cache_key in self.partial_cache:
            return self.partial_cache[cache_key]
                
        return None


    def _get_curvature_points(self, pcd):
        curvatures,_ = compute_curvature_normal(pcd, self.curvature_params['k'])
        partial_points = np.array(pcd.points)
        points_selected_curv = select_points_by_curvature(pcd, curvatures, self.n_keypoints,
                                                            self.curvature_params['neigbor_size'],
                                                            self.curvature_params['curvature_thres'],
                                                            self.curvature_params['reduction_factor'],
                                                            self.curvature_params['zero_out_dist'])
        
        _, fps_selected_points = fps_with_initial_samples(partial_points, points_selected_curv, self.n_keypoints)

        if len(points_selected_curv) == 0:
            return fps_selected_points
        if len(fps_selected_points) == 0:
            return points_selected_curv[:self.n_keypoints,:]
        
        return np.concatenate([points_selected_curv, fps_selected_points]).astype(np.float32)


    def _get_curvature_points_from_region(self, pcd, n_regions, k=30, thres=0.0):
        curvatures,_ = compute_curvature_normal(pcd, k)
        pcd_points = np.array(pcd.points)
        # Select basis points that will serve as regions
        basis_points = deterministic_fps(pcd_points, n_regions)
        # Find distance from each point to each basis
        distances = np.sqrt(np.sum((pcd_points[:, np.newaxis, :] - basis_points[np.newaxis,:,:])**2, axis=-1))
        # Find closest basis point for each point
        closest_basis_indices = np.argsort(distances, axis=-1)[:,0]
        selected_points_coord = np.zeros((n_regions, 3))

        # For each region find the point with the max curvature
        for i in range(n_regions):
            assigned_points = np.where(closest_basis_indices == i)[0]
            assigned_point_curvatures = curvatures[assigned_points]
            # Below index will be in terms of assigned_point_curvatures list
            max_curvature_index = np.argmax(assigned_point_curvatures)
            # If the point curvature is not high, use basis point
            if assigned_point_curvatures[max_curvature_index] < thres:
                selected_points_coord[i] = basis_points[i]
            else:
                max_curvature_points = pcd_points[assigned_points[max_curvature_index]]
                selected_points_coord[i] = max_curvature_points

        return selected_points_coord

    def _get_keypoints(self, category_name, sample_name, view_id):

        if "snake" in self.keypoint_type:
            keypoint_dir = './data/SNAKEPCN/SNAKE8/{}/{}/partial'.format(self.subset, category_name)
        elif "skeleton" in self.keypoint_type:
            keypoint_dir = './data/SkeletonMergerKeypoints{}/{}_partial/{}'.format(self.n_keypoints, self.subset, category_name)

        try:
            if "skeleton" in self.keypoint_type:
                keypoint_name = "{}_0{}.npy".format(sample_name, view_id)

                keypoint_path = os.path.join(keypoint_dir, keypoint_name)

                with open(keypoint_path, "rb") as npfile:
                    # Skip the final confidence dim
                    keypoints = np.load(npfile)[:,:3]
                return keypoints

            keypoint_name = "0{}.npy".format(view_id)
            keypoint_path = os.path.join(keypoint_dir, sample_name, keypoint_name)

            with open(keypoint_path, "rb") as npfile:
                # Skip the final confidence dim
                keypoints = np.load(npfile)[:,:3]
            selected_keypoints = deterministic_fps(keypoints, self.n_keypoints)
        
        except:
            print("Cannot load keypoints from: ", keypoint_path)
            return np.zeros((self.n_keypoints, 3)).astype(np.float32)
                    
        return selected_keypoints  

    def _write_to_cache(self, cache_key, data, data_type):
        
        if self.cache_dir != "":
            item_cache_dir = os.path.join(self.cache_dir, cache_key)
            with open(item_cache_dir, 'wb') as npfile:
                np.save(npfile, data)
        
        # Check memory cache
        elif cache_key not in self.gt_cache and data_type == "gt":
            self.gt_cache[cache_key] = data

        elif cache_key not in self.partial_cache and data_type == "partial":
            self.partial_cache[cache_key] = data

    def __len__(self):
        if len(self.file_list) == 1:
            return len(self.render_idx)
        return len(self.file_list)
    
    def _get_curvature_points_from_radius(self, pcd, pcd_points, n_regions, k=32, radius=0.3):
        curvatures, pcd = compute_curvature_normal(pcd, k)
        basis_points = deterministic_fps(pcd_points, n_regions)
        distances_to_bases = np.sqrt(np.sum((basis_points[:, np.newaxis, :] - pcd_points[np.newaxis,:,:])**2, axis=-1))
        
        # Mark all points out of the radius ball 
        radius_mask = np.where(distances_to_bases > radius, True, False)
        
        selected_keypoints = np.zeros((n_regions, 3))

        for i in range(n_regions):
            temp_curv = curvatures.copy()
            # Reduce all curvatures values to negative to eliminate them  
            temp_curv[radius_mask[i]] = -1.0
            cur_max_curv_point_idx = np.argmax(temp_curv)
            selected_keypoints[i] = pcd_points[cur_max_curv_point_idx]
        
        return selected_keypoints

    
    def _get_desc_curvature_points_from_radius(self, category, sample_name, view_num, pcd, pcd_points, n_regions, k=32, radius=0.3):
    
        basis_points = deterministic_fps(pcd_points, n_regions)

        if desc is None:
            return basis_points

        curvatures, pcd = compute_curvature_descriptor(pcd, desc, k)
        distances_to_bases = np.sqrt(np.sum((basis_points[:, np.newaxis, :] - pcd_points[np.newaxis,:,:])**2, axis=-1))
        
        # Mark all points out of the radius ball 
        radius_mask = np.where(distances_to_bases > radius, True, False)
        
        selected_keypoints = np.zeros((n_regions, 3))

        for i in range(n_regions):
            temp_curv = curvatures.copy()
            # Reduce all curvatures values to negative to eliminate them  
            temp_curv[radius_mask[i]] = -1.0
            cur_max_curv_point_idx = np.argmax(temp_curv)
            selected_keypoints[i] = pcd_points[cur_max_curv_point_idx]
        
        return selected_keypoints
