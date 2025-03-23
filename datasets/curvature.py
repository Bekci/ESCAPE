import open3d as o3d
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from datasets.data_transforms import deterministic_fps


def select_points_by_curvature(pcd, curvatures, n_points, neighbor_size=64, curvature_thres=0.2, reduction_factor=0.5, zero_out_distance=0.05):

    points = np.array(pcd.points)
    num_knn_neighbors = min(points.shape[0], neighbor_size)
    nbrs = NearestNeighbors(n_neighbors=num_knn_neighbors, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    selected_indices = []

    for i in range(n_points):
        cur_index = np.argmax(curvatures)
        if curvatures[cur_index] < curvature_thres:
            break
        
        # Zero out all curvature values of the points which are closer to the selected point as zero_out_distance
        selected_point = points[cur_index]
        distances = euclidean_distances(points, np.expand_dims(selected_point, axis=0))
        close_point_indices = np.where(distances < zero_out_distance)[0]
        curvatures[close_point_indices] = 0.0

        # Reduce the curvature value of the points by a reduction_factor which are close to the selected point
        cur_neighbors =  indices[cur_index, :]
        curvatures[cur_neighbors] = curvatures[cur_neighbors] * reduction_factor
        selected_indices.append(cur_index)    
    
    if len(selected_indices) == 0:
        return []
    return points[selected_indices, :]

def compute_curvature_normal(pcd, k=30):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        centroid = np.mean(np.asarray(pcd.points), axis=0)
        pcd.orient_normals_towards_camera_location(centroid)

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    num_knn_neighbors = min(points.shape[0], k)
    nbrs = NearestNeighbors(n_neighbors=num_knn_neighbors, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    curvatures = np.zeros(len(points))
    for i in range(len(points)):
        neighbors = indices[i]
        neighbor_normals = normals[neighbors]
        covariance_matrix = np.cov(neighbor_normals, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures[i] = abs(curvature)

    curvatures = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min())
    colors = plt.cm.jet(curvatures)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return curvatures, pcd

def compute_curvature_descriptor(pcd, descriptors, k=30):
    k = 32
    points = np.asarray(pcd.points)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    curvatures = np.zeros(len(points))

    for i in range(len(points)):
        neighbors = indices[i]
        neighbor_descs = descriptors[neighbors]
        eigenvalues, _ = np.linalg.eigh(neighbor_descs)
        curvature = eigenvalues[0] / np.sum(eigenvalues)
        curvatures[i] = abs(curvature)

    curvatures = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min())
    colors = plt.cm.jet(curvatures)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return curvatures, pcd

def get_basis_improved_curv_points(pcd, pcd_points, k=32, n_keypoint=8, n_neighbors=32):
    basis_points = deterministic_fps(pcd_points, n_keypoint)
    return get_improved_curv_points_from_initial(pcd, pcd_points, basis_points, k=k, n_neighbors=n_neighbors)


def get_improved_curv_points_from_initial(pcd, pcd_points, initial_points, k=32, n_neighbors=32):
    curvatures, pcd = compute_curvature_normal(pcd, k)
    distances_to_bases = np.sqrt(np.sum((initial_points[:, np.newaxis, :] - pcd_points[np.newaxis,:,:])**2, axis=-1))
    # Closest points to the bases
    indices_basis_closest_n_points = np.argsort(distances_to_bases)[:,:n_neighbors]
    # Among closest points to bases, point indices with max curvatures 
    improved_neigbor_indices = np.argmax(curvatures[indices_basis_closest_n_points],axis=-1)
    # From closest neighbor point indices, find point indices in the input point cloud
    improved_point_indices = indices_basis_closest_n_points[np.arange(indices_basis_closest_n_points.shape[0]), improved_neigbor_indices]
    selected_keypoints = pcd_points[improved_point_indices]
    return selected_keypoints


def get_point_with_max_curv_around(pcd, pcd_points, k=32, n_regions=8, n_neighbors=32):
    curvatures,_ = compute_curvature_normal(pcd, k)
    pcd_points = np.array(pcd.points)
    basis_points = deterministic_fps(pcd_points, n_regions)

    distances = np.sqrt(np.sum((pcd_points[:, np.newaxis, :] - basis_points[np.newaxis,:,:])**2, axis=-1))
    # Find closest basis point for each point
    closest_basis_indices = np.argsort(distances, axis=-1)[:,0]

    # Distance from each point to each others
    p2p_distances = np.sqrt(np.sum((pcd_points[:, np.newaxis, :] - pcd_points[np.newaxis,:,:])**2, axis=-1))
    point_neighbors = np.argsort(p2p_distances, axis=1)[:,:n_neighbors]
    mean_curv_around_points = np.mean(curvatures[point_neighbors],axis=-1, keepdims=True)
    selected_points_coord = np.zeros((n_regions, 3))

    # For each region find the point with the max curvature
    for i in range(n_regions):
        assigned_points = np.where(closest_basis_indices == i)[0]
        assigned_point_curvatures = mean_curv_around_points[assigned_points]
        
        # Below index will be in terms of assigned_point_curvatures list
        max_curvature_index = np.argmax(assigned_point_curvatures)
        max_curvature_points = pcd_points[assigned_points[max_curvature_index]]
        selected_points_coord[i] = max_curvature_points
    return selected_points_coord