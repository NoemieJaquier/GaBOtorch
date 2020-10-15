import torch

from BoManifolds.nested_mappings.nested_spd_utils import projection_from_nested_spd_to_spd, \
    projection_from_spd_to_nested_spd

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def max_eigenvalue_nested_spd_constraint(x_nested_spd, maximum_eigenvalue, projection_matrix,
                                         projection_complement_matrix, bottom_spd_matrix, contraction_matrix):
    """
    This function defines an inequality constraint on the maximum eigenvalue of the nested SPD in function of the
    maximum eigenvalue authorized in the original SPD space.
    To do so, the nested SPD is projected back to the original SPD space and the eigenvalues of the resulting matrix
    are computed.

    Parameters
    ----------
    :param x_nested_spd: low-dimensional nested SPD matrix
    :param maximum_eigenvalue: maximum eigenvalue in the original SPD space
    :param projection_matrix: element of the Grassmann manifold                     (D x d)
    :param projection_complement_matrix: element of the Grassmann manifold          (D x D-d)
        Note that we must have torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :param bottom_spd_matrix: bottom part of the rotated SPD matrix                 (D-d, D-d)
    :param contraction_matrix: matrix whose norm is <=1                             (d x D-d)

    Returns
    -------
    :return: difference between maximum tolerated eigenvalue and maximum eigenvalue of the SPD matrix in the original
        space
    """
    # Project to original SPD space
    x_spd = projection_from_nested_spd_to_spd(x_nested_spd, projection_matrix, projection_complement_matrix,
                                      bottom_spd_matrix, contraction_matrix)
    # Eigenvalue decomposition
    eigenvalues = torch.symeig(x_spd, eigenvectors=True).eigenvalues
    return maximum_eigenvalue - eigenvalues.max()


def min_eigenvalue_nested_spd_constraint(x_nested_spd, minimum_eigenvalue, projection_matrix,
                                         projection_complement_matrix, bottom_spd_matrix, contraction_matrix):
    """
    This function defines an inequality constraint on the minimum eigenvalue of the nested SPD in function of the
    maximum eigenvalue authorized in the original SPD space.
    To do so, the nested SPD is projected back to the original SPD space and the eigenvalues of the resulting matrix
    are computed.

    Parameters
    ----------
    :param x_nested_spd: low-dimensional nested SPD matrix
    :param minimum_eigenvalue: minimum eigenvalue in the original SPD space
    :param projection_matrix: element of the Grassmann manifold                     (D x d)
    :param projection_complement_matrix: element of the Grassmann manifold          (D x D-d)
        Note that we must have torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :param bottom_spd_matrix: bottom part of the rotated SPD matrix                 (D-d, D-d)
    :param contraction_matrix: matrix whose norm is <=1                             (d x D-d)

    Returns
    -------
    :return: difference between minimum eigenvalue of the SPD matrix in the original space and the minimum tolerated
        eigenvalue
    """
    # Project to original SPD space
    x_spd = projection_from_nested_spd_to_spd(x_nested_spd, projection_matrix, projection_complement_matrix,
                                              bottom_spd_matrix, contraction_matrix)
    # Eigenvalue decomposition
    eigenvalues = torch.symeig(x_spd, eigenvectors=True).eigenvalues
    return eigenvalues.min() - minimum_eigenvalue


def random_nested_spd_with_spd_eigenvalue_constraints(self, random_spd_fct, projection_matrix):
    """
    This function computes a nested SPD sample by computing first a sample in the original SPD space and projecting it
    into the nested SPD space.

    Parameters
    ----------
    :param self: self parameter of the SPD pymanopt class
    :param random_spd_fct: function to generate SPD samples in the original space
    :param projection_matrix: element of the Grassmann manifold                     (D x d)

    Returns
    -------
    :return: nested SPD sample
    """
    # Sample a SPD sample respecting the constraint in the original space
    x_spd = torch.tensor(random_spd_fct(), dtype=projection_matrix.dtype)

    # Project it to nested SPD
    nested_spd = projection_from_spd_to_nested_spd(x_spd, projection_matrix)
    # To numpy to stay within pymanopt format
    return nested_spd.numpy()
