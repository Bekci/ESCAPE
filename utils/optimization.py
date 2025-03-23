import numpy as np
from scipy.optimize import least_squares


def jacobian(point, ps, ds):
    """
    Computes the Jacobian matrix of the residuals with respect to the point coordinates.
    Params:
        point: np.array of the unknown point coordinates (1,3).
        ps: np.array of known points (N,3).
        ds: np.array of distances to known points (N,).
    Returns:
        Jacobian matrix with shape (N,3).
    """
    diff = ps - point  # Difference between known points and the unknown point (N, 3)
    distances = np.sqrt(np.sum(diff**2, axis=1))  # Distances from the unknown point to the known points (N,)

    # Avoid division by zero for coincident points
    distances[distances == 0] = 1e-8

    # Jacobian components
    J = -diff / distances[:, np.newaxis]  # Partial derivatives (N, 3)
    return J

# Function that computes residuals (differences between computed distances and given distances)
def residuals(point, ps, ds):
    computed_distances = np.sqrt(np.sum((ps - point)**2, axis=1))
    return computed_distances - ds


def find_points_from_distance(distances, known_points):
    """
    Finds the coordinates of the 3D points using their distances to the known points
    Params:
        distances: np.array of distances of the unknown points to the known points. With the shape of (U,N)
        known_points: np.array of the known points. With the shape of (N,3)
    Returns:
        The 3D point coordinates as np.array with the shape of (U,3) optimized by the least squares problem
    """
    loss_function = 'linear'
    initial_guess = np.array([0.0, 0.0, 0.0])
    jac_method = jacobian
    optimizer = 'lm'
    
    optimized_points = np.zeros((distances.shape[0], 3))

    for i in range(optimized_points.shape[0]):
        point_distances = distances[i]
        # Use the Levenberg-Marquardt solver to find the point (x, y, z)
        result = least_squares(
            residuals,
            initial_guess,
            args=(known_points, point_distances),
            method=optimizer,
            jac=jac_method,
            loss=loss_function
            )
        optimized_points[i] = result.x
    
    return optimized_points