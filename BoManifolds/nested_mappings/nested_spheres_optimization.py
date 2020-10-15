import numpy as np
import torch
import gpytorch

import pymanopt.manifolds as pyman_man

from BoManifolds.pymanopt_addons.problem import Problem

from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch
from BoManifolds.nested_mappings.nested_spheres_utils import projection_from_subsphere_to_sphere

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def min_error_reconstruction_cost(x_data, x_subsphere, sphere_axes, sphere_distances):
    """
    This function computes the squared error between a set of data x on the sphere and their reconstruction from the
    corresponding projection from the nested subsphere.

    Parameters
    ----------
    :param x_data: original data on the sphere S^d
    :param x_subsphere: data on the subsphere S^d-r
    :param sphere_axes: list of nested sphere axes [S^d, S^d-1, ... S^d-r+1]
    :param sphere_distances: list of nested sphere distances to axes [S^d, S^d-1, ... S^d-r+1]

    Returns
    -------
    :return: sum of squared distances between reconstructed and original data on the sphere
    """
    x_reconstructed = projection_from_subsphere_to_sphere(x_subsphere, sphere_axes, sphere_distances)[-1]
    cost = sphere_distance_torch(x_data, x_reconstructed, diag=True)
    return torch.sum(cost * cost)


def optimize_reconstruction_parameters_nested_sphere(x_data, x_subsphere, sphere_axes, solver,
                                                     nb_init_candidates=100):
    """
    This function computes the distances-to-axis parameters of the "projection_from_subsphere_to_sphere", so that the
    distance between the original and reconstructed data on the sphere S^d is minimized.
    The problem is treated as an unconstrained optimization problem on a product of Euclidean manifolds
    by transforming the interval [0,pi] for the distances with a sigmoid function.

    Parameters
    ----------
    :param x_data: original data on the sphere S^d
    :param x_subsphere: data on the subsphere S^d-r
    :param sphere_axes: list of nested sphere axes [S^d, S^d-1, ... S^d-r+1]
    :param solver: optimization solver

    Optional parameters
    -------------------
    :param nb_init_candidates: number of initial candidates for the optimization

    Returns
    -------
    :return: list of nested sphere distances to axes [S^d, S^d-1, ... S^d-r+1]
    """
    # Dimensions
    dim = x_data.shape[1]
    latent_dim = x_subsphere.shape[1]

    # Product of Euclidean manifold
    manifolds_list = [pyman_man.Euclidean(1) for dim in range(dim, latent_dim, -1)]
    product_manifold = pyman_man.Product(manifolds_list)

    # Interval constraint [0,pi] for the distances to axis
    radius_constraint = gpytorch.constraints.Interval(0., np.pi)

    # Define the reconstruction cost
    def reconstruction_cost(parameters):
        sphere_distances = [radius_constraint.transform(p) for p in parameters]
        return min_error_reconstruction_cost(x_data, x_subsphere, sphere_axes, sphere_distances)

    # Generate candidate for initial data
    x0_candidates = [product_manifold.rand() for i in range(nb_init_candidates)]
    x0_candidates_torch = []
    for x0 in x0_candidates:
        x0_candidates_torch.append([torch.from_numpy(x) for x in x0])
    y0_candidates = [reconstruction_cost(x0_candidates_torch[i]) for i in range(nb_init_candidates)]

    # Initialize with the best of the candidates
    y0, x_init_idx = torch.Tensor(y0_candidates).min(0)
    x0 = x0_candidates[x_init_idx]

    # Define the problem
    reconstruction_problem = Problem(manifold=product_manifold, cost=reconstruction_cost, arg=torch.Tensor(),
                                     verbosity=0)
    # Solve
    sphere_parameters_np = solver.solve(reconstruction_problem, x=x0)

    # Return torch data
    return [radius_constraint.transform(torch.Tensor(distance)) for distance in sphere_parameters_np]
