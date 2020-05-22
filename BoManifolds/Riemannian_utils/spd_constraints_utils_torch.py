import numpy as np
import torch

from BoManifolds.Riemannian_utils.spd_utils_torch import vector_to_symmetric_matrix_mandel_torch, \
    symmetric_matrix_to_vector_mandel_torch

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def max_eigenvalue_constraint_torch(x, maximum_eigenvalue):
    """
    This function defines an inequality constraint on the maximum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x: SPD matrix
    :param maximum_eigenvalue: maximum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between maximum tolerated eigenvalue and maximum eigenvalue of x
    """
    eigenvalues = torch.symeig(x, eigenvectors=True).eigenvalues  # Eigenvalue True necessary for derivation
    return maximum_eigenvalue - eigenvalues.max()


def min_eigenvalue_constraint_torch(x, minimum_eigenvalue):
    """
    This function defines an inequality constraint on the minimum eigenvalue of a SPD matrix.
    The value returned by the function is positive if the inequality constraints is satisfied.

    Parameters
    ----------
    :param x: SPD matrix
    :param minimum_eigenvalue: minimum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: difference between minimum eigenvalue of x and minimum tolerated eigenvalue
    """
    eigenvalues = torch.symeig(x, eigenvectors=True).eigenvalues  # Eigenvalue True necessary for derivation
    return eigenvalues.min() - minimum_eigenvalue


def post_processing_init_spd_torch(x_vec, min_eigenvalue, max_eigenvalue):
    """
    This function post-processes a symmetric matrix, so that its eigenvalues lie in the defined bounds.

    Parameters
    ----------
    :param x_vec: symmetric matrix in Mandel vector form
    :param min_eigenvalue: minimum eigenvalue to satisfy the constraint
    :param max_eigenvalue: maximum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: symmetric matrix in Mandel vector form
    """
    x = vector_to_symmetric_matrix_mandel_torch(x_vec)
    init_shape = x.shape
    x = x.view(-1, x.shape[-2], x.shape[-1])

    for n in range(x.shape[0]):
        eigdec = torch.symeig(x[n], eigenvectors=True)
        eigvals = eigdec.eigenvalues
        eigvecs = eigdec.eigenvectors

        eigvals[eigvals <= min_eigenvalue] = min_eigenvalue  # Minimum eigenvalue constraint
        eigvals[eigvals > max_eigenvalue] = max_eigenvalue  # Max eigenvalue constraint

        x[n] = torch.mm(torch.mm(eigvecs, torch.diag(eigvals)), torch.inverse(eigvecs))

    x = x.view(init_shape)
    return symmetric_matrix_to_vector_mandel_torch(x)


def post_processing_spd_cholesky_torch(x_chol, min_eigenvalue, max_eigenvalue):
    """
    This function post-processes a symmetric matrix, so that its eigenvalues lie in the defined bounds.

    Parameters
    ----------
    :param x_vec: symmetric matrix in Mandel vector form
    :param min_eigenvalue: minimum eigenvalue to satisfy the constraint
    :param max_eigenvalue: maximum eigenvalue to satisfy the constraint

    Returns
    -------
    :return: symmetric matrix in Mandel vector form
    """
    # Initial shape
    init_shape = list(x_chol.shape)
    x_chol = x_chol.view(-1, init_shape[-1])

    # Dimension of SPD matrix
    dim_vec = x_chol.shape[-1]
    dim = int((-1.0 + (1.0 + 8.0 * dim_vec) ** 0.5) / 2.0)
    # Indices for Cholesky decomposition
    indices = np.tril_indices(dim)

    for n in range(x_chol.shape[0]):
        # SPD matrix
        xL = torch.zeros((dim, dim), dtype=x_chol.dtype)
        xL[indices] = x_chol[n]
        x_mat = torch.mm(xL, xL.T)

        # Check constraints
        eigdec = torch.eig(x_mat, eigenvectors=True)
        eigvals = eigdec.eigenvalues[:, 0]
        eigvecs = eigdec.eigenvectors

        eigvals[eigvals <= min_eigenvalue] = min_eigenvalue  # PD constraint
        eigvals[eigvals > max_eigenvalue] = max_eigenvalue  # Max eigenvalue constraint

        x_mat = torch.mm(torch.mm(eigvecs, torch.diag(eigvals)), torch.inverse(eigvecs))

        # Cholesky decomposition
        x_chol[n] = torch.cholesky(x_mat)[indices]

    return x_chol.view(init_shape)

