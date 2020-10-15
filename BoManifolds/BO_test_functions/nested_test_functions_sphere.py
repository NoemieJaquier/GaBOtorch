import torch

from BoManifolds.nested_mappings.nested_spheres_utils import projection_from_sphere_to_subsphere, \
    projection_from_subsphere_to_sphere

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def nested_function_sphere(x, subsphere_manifold, test_function, sphere_axes, sphere_distances_to_axes):
    """
    This function computes the value of a test function defined on a subsphere S^m of the original sphere manifold
    S^d.

    Note: test functions and their global minimum are defined in test_function_sphere.py.

    Parameters
    ----------
    :param x: point on the sphere                                   (torch tensor)
    :param test_function: test function defined on the subsphere S^m
    :param subsphere_manifold: m-dimensional sphere manifold        (pymanopt manifold)
    :param sphere_axes: axes of the nested spheres belonging to [Sd, Sd-1, ..., Sm+1]
    :param sphere_distances_to_axes: distances from the axes w.r.t each point of the nested spheres of
    [Sd, Sd-1, ..., Sm+1]

    Returns
    -------
    :return: value of the test function at x                        (numpy [1,1] array)
    """
    # Two-dimensional input
    if x.dim() < 2:
        x = x[None]
    # Projection into subsphere
    x_subsphere = projection_from_sphere_to_subsphere(x, sphere_axes, sphere_distances_to_axes)[-1]

    # Compute the test function value
    return test_function(x_subsphere, subsphere_manifold)


def optimum_nested_function_sphere(optimum_function, subsphere_manifold, sphere_axes, sphere_distances_to_axes):
    """
    This function returns the global minimum (x, f(x)) of a test function defined on a subsphere S^m of the original
    sphere manifold S^d.
    Note that the location of the global minimum is unique on the subsphere but not on the original sphere. All the
    locations on the sphere that are projected onto the minimum point on the nested sphere are minimum of the function.
    We return here the minimum point of the sphere belonging directly to the nested sphere.

    Note: test functions and their global minimum are defined in test_function_sphere.py.

    Parameters
    ----------
    :param optimum_function: function returning the global minimum (x, f(x)) of the test function on the subsphere
    :param subsphere_manifold: m-dimensional sphere manifold        (pymanopt manifold)
    :param sphere_axes: axes of the nested spheres belonging to [Sd, Sd-1, ..., Sd-r+1]
    :param sphere_distances_to_axes: distances from the axes w.r.t each point of the nested spheres of
    [Sd, Sd-1, ..., Sd-r+1]

     Returns
    -------
    :return opt_x: location of the global minimum of the test function on the sphere
    :return opt_y: value of the global minimum of the test function on the sphere
    """
    # Global minimum on subsphere
    nested_opt_x, opt_y = optimum_function(subsphere_manifold)
    # To torch
    nested_opt_x_torch = torch.tensor(nested_opt_x, dtype=sphere_axes[0].dtype)

    # Projection onto the original sphere space
    opt_x_torch = projection_from_subsphere_to_sphere(nested_opt_x_torch, sphere_axes, sphere_distances_to_axes)[-1]

    # To numpy
    opt_x = opt_x_torch.numpy()

    return opt_x, opt_y


