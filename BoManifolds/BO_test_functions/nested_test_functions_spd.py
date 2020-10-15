import numpy as np
import torch

from BoManifolds.Riemannian_utils.spd_utils import symmetric_matrix_to_vector_mandel
from BoManifolds.Riemannian_utils.spd_utils_torch import vector_to_symmetric_matrix_mandel_torch, \
    symmetric_matrix_to_vector_mandel_torch
from BoManifolds.nested_mappings.nested_spd_utils import projection_from_spd_to_nested_spd

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def projected_function_spd(x, low_dimensional_spd_manifold, test_function, projection_matrix):
    """
    This function computes the value of a test function defined on a projection of the original SPD manifold S^D_++
    on a lower-dimensional SPD manifold S^d_++.

    Note: test functions and their global minimum are defined in test_function_spd.py.

    Parameters
    ----------
    :param x: point on the SPD manifold (in Mandel notation)                        (torch tensor)
    :param low_dimensional_spd_manifold: d-dimensional SPD manifold                 (pymanopt manifold)
    :param test_function: test function defined on the low-dimensional SPD manifold
    :param projection_matrix: element of the Grassmann manifold                     (D x d)


    Returns
    -------
    :return: value of the test function at x                        (numpy [1,1] array)
    """
    # Mandel to matrix
    x_spd = vector_to_symmetric_matrix_mandel_torch(x)

    # Projection to lower dimensional SPD manifold
    x_spd_low_dimension = projection_from_spd_to_nested_spd(x_spd, projection_matrix)

    # From matrix to Mandel
    x_low_dimension = symmetric_matrix_to_vector_mandel_torch(x_spd_low_dimension)

    # Compute the test function value
    return test_function(x_low_dimension, low_dimensional_spd_manifold)


def optimum_projected_function_spd(optimum_function, low_dimensional_spd_manifold, projection_matrix):
    """
    This function returns the global minimum (x, f(x)) of a test function defined on a projection of a SPD manifold
    S^D_++ in a lower dimensional SPD manifold S^d_++.
    Note that, as the inverse of the projection function is not available, the location of the optimum x on the
    low-dimensional manifold is returned.

    Note: test functions and their global minimum are defined in test_function_spd.py.

    Parameters
    ----------
    :param optimum_function: function returning the global minimum (x, f(x)) of the test function on S^d_++
    :param low_dimensional_spd_manifold: d-dimensional SPD manifold                 (pymanopt manifold)
    :param projection_matrix: element of the Grassmann manifold                     (D x d)

    Returns
    -------
    :return opt_x: location of the global minimum of the Ackley function on the low-dimensional SPD manifold
    :return opt_y: value of the global minimum of the Ackley function on the SPD manifold
    """
    # Global minimum on nested SPD
    nested_opt_x, opt_y = optimum_function(low_dimensional_spd_manifold)

    return nested_opt_x, opt_y


def cholesky_embedded_function_wrapped(x_cholesky, low_dimensional_spd_manifold, spd_manifold, test_function):
    """
    This function is a wrapper for tests function on the SPD manifold with inputs in the form of a Cholesky
    decomposition. The Cholesky decomposition input is transformed into the corresponding SPD matrix which is then
    given as input for the given test function.

    Parameters
    ----------
    :param x_cholesky: Cholesky decomposition of a SPD matrix
    :param low_dimensional_spd_manifold: d-dimensional SPD manifold     (pymanopt manifold)
    :param spd_manifold: D-dimensional SPD manifold                     (pymanopt manifold)
    :param test_function: embedded function on the low-dimensional SPD manifold to be tested

    Returns
    -------
    :return: value of the test function at x                            (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Data to numpy
    torch_type = x_cholesky.dtype
    x_cholesky = x_cholesky.detach().numpy()

    if np.ndim(x_cholesky) == 2:
        x_cholesky = x_cholesky[0]

    # Verify that Cholesky decomposition does not have zero
    if x_cholesky.size - np.count_nonzero(x_cholesky):
        x_cholesky += 1e-6

    # Add also a small value to too-close-to-zero Cholesky decomposition elements
    x_cholesky[np.abs(x_cholesky) < 1e-10] += 1e-10

    # Reshape matrix
    indices = np.tril_indices(dimension)
    xL = np.zeros((dimension, dimension))
    xL[indices] = x_cholesky

    # Compute SPD from Cholesky
    x = np.dot(xL, xL.T)
    # Mandel notation
    x = symmetric_matrix_to_vector_mandel(x)
    # To torch
    x = torch.from_numpy(x).to(dtype=torch_type)

    # Test function
    return test_function(x, low_dimensional_spd_manifold)
