import h5py
import numpy as np
import open3d
import os

import cv2
import math
import numpy as np
import torch
import random
import transforms3d

def fps_with_initial_samples(points, initial_points, npoint):
    """
    Input:
        points: pointcloud data, [N, D]
        initial_points: already sampled points from 'points'
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N,_ = points.shape

    # If the initial set is empty, sample all with fps
    if len(initial_points) == 0:
        fps_sampled_points = deterministic_fps(points, npoint)
        return initial_points, fps_sampled_points
    
    n_point_exist, _ = initial_points.shape
    
    # Enough points are selected
    if n_point_exist >= npoint:
        return initial_points[:npoint, :], np.array([])

    remaining_points = npoint - n_point_exist

    # Run FPS with initial set of points for remaining points
    distance = np.ones((N,)) * 1e10

    # First iterate through all existing points to fill distance values
    for i in range(0, n_point_exist):
        centroid = initial_points[i, :]
        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]

    # Select remaining points with FPS
    centroids = np.zeros((remaining_points,))
    farthest = np.argmax(distance, -1)

    for i in range(remaining_points):
        centroids[i] = farthest
        centroid = points[farthest, :]

        dist = np.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    fps_selected_points = points[centroids.astype(np.int32)]
    return initial_points, fps_selected_points


def batched_deterministic_fps(point, npoint):
    """
    Input:
        point: point cloud data, [B, N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint, D]
    """
    B, N, D = point.shape
    xyz = point[:, :, :3]
    
    centroids = np.zeros((B, npoint, D), dtype=point.dtype)
    
    for b in range(B):
        distance = np.ones((N,)) * 1e10
        center_point = np.mean(point[b], axis=0)
        dist_to_center = np.sum((xyz[b] - center_point) ** 2, -1)
        farthest = np.argmax(dist_to_center, -1)
        
        for i in range(npoint):
            centroids[b, i] = point[b, farthest]

            centroid = xyz[b, farthest, :]
            dist = np.sum((xyz[b] - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
    
    return centroids


def deterministic_fps(point, npoint):
    """

    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10

    center_point = np.mean(point, axis=0)
    
    dist_to_center = np.sum((xyz - center_point) ** 2, -1)
    farthest = np.argmax(dist_to_center, -1)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]

        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                            RandomCrop, RandomFlip, RandomRotatePoints, ScalePoints, RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:  # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, arr):
        arr = arr.astype(np.float32)
        arr /= self.std
        arr -= self.mean

        return arr


class CenterCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * .5
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * .5
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img, rnd_value):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * rnd_value
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * rnd_value
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img, rnd_value):
        if rnd_value > 0.5:
            img = np.fliplr(img)

        return img


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img):
        rgb_permutation = np.random.permutation(3)
        return img[..., rgb_permutation]


class RandomBackground(object):
    def __init__(self, parameters):
        self.random_bg_color_range = parameters['bg_color']

    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        if not img_c == 4:
            return img

        r, g, b = [
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]]) / 255.
        img = alpha * bg_color + (1 - alpha) * img

        return img


class SelectPoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need == 0:
            return ptcloud
        
        while need > 0:
            ptcloud = np.tile(ptcloud, (2, 1))
            curr = ptcloud.shape[0]
            need = self.n_points - curr
        
        ptcloud = deterministic_fps(ptcloud, self.n_points)
        return ptcloud



class UpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        n_valid = random.randint(512, 1024)
        ptcloud = farthest_point_sample(ptcloud, n_valid)
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class DownUpSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        curr = ptcloud.shape[0]
        need = self.n_points - curr

        if need < 0:
            return ptcloud[np.random.permutation(self.n_points)]

        while curr <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= curr
            curr *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

        return ptcloud


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomClipPoints(object):
    def __init__(self, parameters):
        self.sigma = parameters['sigma'] if 'sigma' in parameters else 0.01
        self.clip = parameters['clip'] if 'clip' in parameters else 0.05

    def __call__(self, ptcloud):
        ptcloud += np.clip(self.sigma * np.random.randn(*ptcloud.shape), -self.clip, self.clip).astype(np.float32)
        return ptcloud


class RandomRotatePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        angle = 2 * math.pi * rnd_value
        trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class ScalePoints(object):
    def __init__(self, parameters):
        self.scale = None
        if 'scale' in parameters:
            self.scale = parameters['scale']

    def __call__(self, ptcloud, rnd_value):
        if self.scale is not None:
            scale = self.scale
        else:
            scale = np.random.randint(85, 95) * 0.01
        ptcloud = ptcloud * scale
        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)


        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.pcd']:
            return cls._write_pcd(file_path, file_content)
        elif file_extension in ['.h5']:
            return cls._write_h5(file_path, file_content)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_pcd(cls, file_path):

        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=file_content)
