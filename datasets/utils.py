import open3d as o3d
import numpy as np
from easydict import EasyDict as edict
import yaml
from sklearn import decomposition

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals


def construct_inputs(point_tensor):
    np_points = point_tensor.numpy()
    o3d_src_pcd = to_o3d_pcd(np_points)
    
    o3d_src_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
    
    o3d_src_pcd.orient_normals_to_align_with_direction()
    src_normals = np.asarray(o3d_src_pcd.normals)

    src_o = np.array([np_points.shape[0]], dtype=np.int32)
    src_feats = np.ones(shape=(np_points.shape[0], 1))
    return np_points.astype(np.float32), src_feats.astype(np.float32), src_o, src_normals.astype(np.float32)


def batched_calculate_normals(keypoints):

    b,n,_ = keypoints.shape
    normals = np.zeros_like(keypoints)
    
    for i  in range(b):
        keypoint_pcd = to_o3d_pcd(keypoints[i])
        
        keypoint_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
        
        keypoint_pcd.orient_normals_to_align_with_direction()
        src_normals = np.asarray(keypoint_pcd.normals)
        normals[i] = src_normals

    return normals.astype(np.float32)


def calculate_normals(keypoints):

    keypoint_pcd = to_o3d_pcd(keypoints)
    
    keypoint_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=33))
    
    keypoint_pcd.orient_normals_to_align_with_direction()
    src_normals = np.asarray(keypoint_pcd.normals)

    return src_normals.astype(np.float32)


def read_pcd_fix_size(file_path, fixed_size, deformation=None):
    pc = o3d.io.read_point_cloud(file_path)
    
    if deformation:
        pc = deformation(pc)
    
    ptcloud = np.array(pc.points)
    # Downsample pcd
    if len(ptcloud) > fixed_size:
        down_sampled_pc = pc.farthest_point_down_sample(fixed_size)
        return np.array(down_sampled_pc.points)
    
    # Resample same points in the pc as many as we need and then apply downsampling
    if len(ptcloud) < fixed_size:
        cur = ptcloud.shape[0]
        need = fixed_size - cur        

        while cur <= need:
            ptcloud = np.tile(ptcloud, (2, 1))
            need -= cur
            cur *= 2

        choice = np.random.permutation(need)
        ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

    return ptcloud

def apply_rotation(point_cloud, axis, angle):
    '''
    Apply rotation to given point_cloud on the given axis 
    by the given angle. Axis order (0,1,2) -> (x,y,z)
    '''
    rotation_list = [0,0,0]
    rotation_list[axis] = angle
  
    R = point_cloud.get_rotation_matrix_from_xyz(rotation_list)
    point_cloud.rotate(R, [0,0,0])

    return point_cloud

def apply_translation(point_cloud, translation_vector):
    '''
    Apply translation vector to given point_cloud 
    '''
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = point_cloud.points
    temp_pcd.translate(translation_vector)

    return temp_pcd

def read_pcd(file_path, deformation=None):
    pc = o3d.io.read_point_cloud(file_path)
    
    if deformation:
        pc = deformation(pc)
    
    ptcloud = np.array(pc.points)

    return ptcloud

def to_o3d_pcd(pcd):
    '''
    Transfer a point cloud of numpy.ndarray to open3d point cloud
    :param pcd: point cloud of numpy.ndarray in shape[N, 3]
    :return: open3d.geometry.PointCloud()
    '''
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(pcd)
    return pcd_

def normal_redirect(points, normals, view_point):
    '''
    Make direction of normals towards the view point
    '''
    vec_dot = np.sum((view_point - points) * normals, axis=-1)
    mask = (vec_dot < 0.)
    redirected_normals = normals.copy()
    redirected_normals[mask] *= -1.
    return redirected_normals

def create_edict(pack):
    d = edict()
    for key, value in pack.items():
        if isinstance(value, dict):
            d[key] = create_edict(value)
        else:
            d[key] = value
    return d

def read_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return create_edict(config)



def fit_normalization(partial_pcd, center_point):
    """
    Applies normalization to given point cloud and returns used
    parameter for normalizing other inputs
    """
    positions = np.asarray(partial_pcd.points)
    patch_temp = o3d.geometry.PointCloud()
    copy_pc(patch_temp, partial_pcd)
    
    partial_pcd.translate(-center_point)
    scale = np.max(np.abs(positions))
    positions = positions / scale
    pca = decomposition.PCA(3)
    
    pca.fit(positions)
    transformed_positions = pca.transform(positions)
    
    scale = np.ptp(transformed_positions)
    transformed_positions = transformed_positions / scale
    # Normal calculation
    patch_norm = o3d.geometry.PointCloud()
    patch_norm.points = o3d.utility.Vector3dVector(transformed_positions)    

    return patch_norm, pca, scale


def transform_pcd(pcd, center_point, pca):
    """
    Given point cloud and pca, applied normalization 
    using pca to point cloud
    """    
    positions = np.asarray(pcd.points)
    gt_temp = o3d.geometry.PointCloud()
    copy_pc(gt_temp, pcd)
    
    pcd.translate(-center_point)
    scale = np.max(np.abs(positions))
    positions = positions / scale
    transformed_positions = pca.transform(positions)

    scale = np.ptp(transformed_positions)
    transformed_positions = transformed_positions / scale
    # Normal calculation
    normed_pcd = o3d.geometry.PointCloud()
    normed_pcd.points = o3d.utility.Vector3dVector(transformed_positions)

    return normed_pcd

def transform_points(points, center_point, pca):
    """
    Given points and pca, applied normalization 
    using pca to points
    """        
    points = points - center_point
    
    scale = np.max(np.abs(points))
    points = points / scale
    
    transformed_keypoints = pca.transform(points)

    scale = np.ptp(transformed_keypoints)
    transformed_keypoints = transformed_keypoints / scale

    return transformed_keypoints



def normalize_point(points_pc, center_point):
    """
    Normalizes points in the point cloud with respect to a center
    """
    positions = np.asarray(points_pc.points)
    patch_temp = o3d.geometry.PointCloud()
    copy_pc(patch_temp, points_pc)
    points_pc.translate(-center_point)

    scale = np.max(np.abs(positions))
    positions = positions / scale
    pca = decomposition.PCA(3)

    pca.fit(positions)
    transformed_positions = pca.transform(positions)

    scale = np.ptp(transformed_positions)
    transformed_positions = transformed_positions / scale
    # Normal calculation
    patch_norm = o3d.geometry.PointCloud()
    patch_norm.points = o3d.utility.Vector3dVector(transformed_positions)
    return patch_norm

def copy_pc(output, point_cloud):
    output.colors = point_cloud.colors
    output.normals = point_cloud.normals
    output.points = point_cloud.points
    return True    


def sort_point_clockwise(points):
    """
    Sorts given points in an clockwise order to their center
    """
    # Find the Center of Mass: data is a numpy array of shape (Npoints, 2)
    points2d = np.stack([points[:, 0], points[:, 2]], axis=-1)
    mean = np.mean(points2d, axis=0)
    #mean = np.array([0., 0.])
    # Compute angles
    angles = np.arctan2((points2d-mean)[:, 1], (points2d-mean)[:, 0])
    # Transform angles from [-pi,pi] -> [0, 2*pi]
    angles[angles < 0] = angles[angles < 0] + 2 * np.pi
    # Sort
    sorting_indices = np.argsort(angles)
    sorted_data = points[sorting_indices]
    return sorted_data    


def read_txt(file_path):
    return np.loadtxt(file_path)

def read_npy(file_path):
    return np.load(file_path)