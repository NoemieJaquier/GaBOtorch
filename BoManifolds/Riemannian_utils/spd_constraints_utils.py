import numpy as np

from BoManifolds.Riemannian_utils.spd_utils import vector_to_symmetric_matrix_mandel

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def min_eigenvalue_constraint(x_vec, min_eigenvalue):
    """
    This function defines an inequality constraint on the minimum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x_vec: SPD matrix in Mandel vector form
    :param min_eigenvalue: minimum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between minimum eigenvalue of x and minimum tolerated eigenvalue
    """
    x = vector_to_symmetric_matrix_mandel(x_vec)
    eigenvalues = np.linalg.eigvals(x)
    return np.min(eigenvalues) - min_eigenvalue


def max_eigenvalue_constraint(x_vec, max_eigenvalue):
    """
    This function defines an inequality constraint on the maximum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x_vec: SPD matrix in Mandel vector form
    :param max_eigenvalue: maximum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between maximum tolerated eigenvalue and maximum eigenvalue of x
    """
    x = vector_to_symmetric_matrix_mandel(x_vec)
    eigenvalues = np.linalg.eigvals(x)
    return max_eigenvalue - np.max(eigenvalues)


def min_eigenvalue_constraint_cholesky(x_chol, min_eigenvalue):
    """
    This function defines an inequality constraint on the minimum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x_chol: cholesky decomposition of a SPD matrix in vector form
    :param min_eigenvalue: minimum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between minimum eigenvalue of x and minimum tolerated eigenvalue
    """
    dim_vec = x_chol.shape[0]
    dim = int((-1.0 + (1.0 + 8.0 * dim_vec) ** 0.5) / 2.0)
    indices = np.tril_indices(dim)
    xL = np.zeros((dim, dim))
    xL[indices] = x_chol

    x_mat = np.dot(xL, xL.T)
    eigenvalues = np.linalg.eigvals(x_mat)
    return np.max(eigenvalues) - min_eigenvalue


def max_eigenvalue_constraint_cholesky(x_chol, max_eigenvalue):
    """
    This function defines an inequality constraint on the maximum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x_chol: cholesky decomposition of a SPD matrix in vector form
    :param max_eigenvalue: maximum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between maximum tolerated eigenvalue and maximum eigenvalue of x
    """
    dim_vec = x_chol.shape[0]
    dim = int((-1.0 + (1.0 + 8.0 * dim_vec) ** 0.5) / 2.0)
    indices = np.tril_indices(dim)
    xL = np.zeros((dim, dim))
    xL[indices] = x_chol

    x_mat = np.dot(xL, xL.T)
    eigenvalues = np.linalg.eigvals(x_mat)
    return max_eigenvalue - np.max(eigenvalues)



