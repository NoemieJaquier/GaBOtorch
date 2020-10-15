import torch
import gpytorch

import pymanopt.manifolds as pyman_man

from BoManifolds.pymanopt_addons.problem import Problem

from BoManifolds.manifold_optimization.augmented_Lagrange_method import AugmentedLagrangeMethod

from BoManifolds.Riemannian_utils.spd_utils_torch import affine_invariant_distance_torch, frobenius_distance_torch, \
    logm_torch

from BoManifolds.nested_mappings.nested_spd_utils import projection_from_nested_spd_to_spd

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def min_affine_invariant_distance_reconstruction_cost(x_data, x_data_projected, projection_matrix,
                                                      projection_complement_matrix,
                                                      bottom_spd_matrix, contraction_matrix):
    """
    This function computes the squared error between a set of SPD data X and their reconstruction from the
    corresponding projections Y = W'XW with W \in G(D,d).

    Parameters
    ----------
    :param x_data: set of high-dimensional SPD matrices                                             (N x D x D)
    :param x_data_projected: set of low-dimensional SPD matrices (projected from x_data)            (N x d x d)
    :param projection_matrix: element of the Grassmann manifold                                     (D x d)
    :param projection_complement_matrix: element of the Grassmann manifold                          (D x D-d)
        Note that we must have torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :param bottom_spd_matrix: bottom-right part of the rotated SPD matrix                           (D-d, D-d)
    :param contraction_matrix: matrix whose norm is <=1                                             (d x D-d)

    Returns
    -------
    :return: sum of squared distances between reconstructed and original SPD data
    """
    n_data = x_data_projected.shape[0]
    # Projection from low-dimensional to high-dimensional SPD data
    x_reconstructed = projection_from_nested_spd_to_spd(x_data_projected, projection_matrix,
                                                        projection_complement_matrix, bottom_spd_matrix,
                                                        contraction_matrix)

    # Compute distances between original and reconstructed data
    cost = torch.zeros(n_data)
    for n in range(n_data):
        cost[n] = affine_invariant_distance_torch(x_data[n].unsqueeze(0), x_reconstructed[n].unsqueeze(0))

    # Sum of squared distances
    return torch.sum(cost * cost)


def min_log_euclidean_distance_reconstruction_cost(x_data, x_data_projected, projection_matrix,
                                                   projection_complement_matrix, bottom_spd_matrix, contraction_matrix):
    """
    This function computes the squared error between a set of SPD data X and their reconstruction from the
    corresponding projections Y = W'XW with W \in G(D,d).

    Parameters
    ----------
    :param x_data: set of high-dimensional SPD matrices                                             (N x D x D)
    :param x_data_projected: set of low-dimensional SPD matrices (projected from x_data)            (N x d x d)
    :param projection_matrix: element of the Grassmann manifold                                     (D x d)
    :param projection_complement_matrix: element of the Grassmann manifold                          (D x D-d)
        Note that we must have torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :param bottom_spd_matrix: bottom-right part of the rotated SPD matrix                           (D-d, D-d)
    :param contraction_matrix: matrix whose norm is <=1                                             (d x D-d)

    Returns
    -------
    :return: sum of squared distances between reconstructed and original SPD data
    """
    n_data = x_data_projected.shape[0]
    # Projection from low-dimensional to high-dimensional SPD data
    x_reconstructed = projection_from_nested_spd_to_spd(x_data_projected, projection_matrix,
                                                        projection_complement_matrix, bottom_spd_matrix,
                                                        contraction_matrix)

    # Compute distances between original and reconstructed data
    cost = torch.zeros(n_data)
    for n in range(n_data):
        cost[n] = frobenius_distance_torch(logm_torch(x_data[n]).unsqueeze(0),
                                           logm_torch(x_reconstructed[n]).unsqueeze(0))

    # Sum of squared distances
    return torch.sum(cost * cost)


def optimize_reconstruction_parameters_nested_spd(x_data, x_data_projected, projection_matrix, inner_solver,
                                                  cost_function=min_affine_invariant_distance_reconstruction_cost,
                                                  nb_init_candidates=100, maxiter=50):
    """
    This function computes the parameters of the mapping "projection_from_nested_spd_to_spd" from nested SPD matrices
    Y = W'XW to SPD matrices Xrec, so that the distance between the original data X and the reconstructed data Xrec
    is minimized.
    To do so, we consider that the nested SPD matrix Y = W'XW is the d x d upper-left part of the rotated matrix
    Xr = R'XR, where R = [W, V] and Xr = [Y B; B' C].
    In order to recover X, we assume a constant SPD matrix C, and B = Y^0.5*K*C^0.5 to ensure the PDness of Xr, with
    K a contraction matrix (norm(K) <=1). We first reconstruct Xr, and then Xrec as X = RXrR'.
    We are minimizing the squared distance between X and Xrec, by optimizing the complement to the projection matrix V,
    the bottom SPD matrix C, and the contraction matrix K. The contraction matrix K is described here as a norm-1
    matrix multiplied by a factor in [0,1] (unconstraintly optimized by transform it with a sigmoid function).
    The augmented Lagrange optimization method on Riemannian manifold is used to optimize the parameters on the product
    of manifolds G(D,D-d), SPD(D-d), S(d*(D-d)) and Eucl(1), while respecting the constraint W'V = 0.

    Parameters
    ----------
    :param x_data: set of high-dimensional SPD matrices                                             (N x D x D)
    :param x_data_projected: set of low-dimensional SPD matrices (projected from x_data)            (N x d x d)
    :param projection_matrix: element of the Grassmann manifold                                     (D x d)
    :param inner_solver: inner solver for the ALM on Riemnannian manifolds

    Optional parameters
    -------------------
    :param nb_init_candidates: number of initial candidates for the optimization
    :param maxiter: maximum iteration of ALM solver

    Returns
    -------
    :return: projection_complement_matrix: element of the Grassmann manifold                        (D x D-d)
        so that torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :return: bottom_spd_matrix: bottom-right part of the rotated SPD matrix                         (D-d, D-d)
    :return: contraction_matrix: matrix whose norm is <=1                                           (d x D-d)
    """
    # Dimensions
    dim = x_data.shape[1]
    latent_dim = projection_matrix.shape[1]

    # Product of manifolds for the optimization
    manifolds_list = [pyman_man.Grassmann(dim, dim - latent_dim), pyman_man.PositiveDefinite(dim - latent_dim),
                      pyman_man.Sphere(latent_dim * (dim - latent_dim)), pyman_man.Euclidean(1)]
    product_manifold = pyman_man.Product(manifolds_list)

    # Constraint on the norm of the contraction matrix
    contraction_norm_constraint = gpytorch.constraints.Interval(0., 1.)

    # Constraint W'V = 0
    def constraint_fct(parameters):
        cost = torch.norm(torch.mm(parameters[0].T, projection_matrix))
        zero_element_needed_for_correct_grad = 0. * torch.norm(parameters[1]) + 0. * torch.norm(parameters[2]) + \
                                               0. * torch.norm(parameters[3])
        return cost + zero_element_needed_for_correct_grad

    # Reconstruction cost
    def reconstruction_cost(parameters):
        projection_complement_matrix = parameters[0]
        bottom_spd_matrix = parameters[1]
        contraction_norm = contraction_norm_constraint.transform(parameters[3])
        contraction_matrix = contraction_norm * parameters[2].view(latent_dim, dim-latent_dim)

        return cost_function(x_data, x_data_projected, projection_matrix, projection_complement_matrix,
                                             bottom_spd_matrix, contraction_matrix)

    # Generate candidate for initial data
    x0_candidates = [product_manifold.rand() for i in range(nb_init_candidates)]
    x0_candidates_torch = []
    for x0 in x0_candidates:
        x0_candidates_torch.append([torch.from_numpy(x) for x in x0])
    y0_candidates = [reconstruction_cost(x0_candidates_torch[i]) for i in range(nb_init_candidates)]

    # Initialize with the best of the candidates
    y0, x_init_idx = torch.Tensor(y0_candidates).min(0)
    x0 = x0_candidates[x_init_idx]

    # Define the optimization problem
    reconstruction_problem = Problem(manifold=product_manifold, cost=reconstruction_cost, arg=torch.Tensor(),
                                     verbosity=0)
    # Define ALM solver
    solver = AugmentedLagrangeMethod(maxiter=maxiter, inner_solver=inner_solver, lambdas_fact=0.05)

    # Solve
    spd_parameters_np = solver.solve(reconstruction_problem, x=x0, eq_constraints=constraint_fct)

    # Parameters to torch data
    projection_complement_matrix = torch.from_numpy(spd_parameters_np[0])
    bottom_spd_matrix = torch.from_numpy(spd_parameters_np[1])
    contraction_norm = contraction_norm_constraint.transform(torch.from_numpy(spd_parameters_np[3]))
    contraction_matrix = contraction_norm * torch.from_numpy(spd_parameters_np[2]).view(latent_dim, dim-latent_dim)

    return projection_complement_matrix, bottom_spd_matrix, contraction_matrix
