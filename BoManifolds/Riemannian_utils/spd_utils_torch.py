import numpy as np
import torch
'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def logm_torch(x):
    """
    This function computes the logarithm of a matrix.

    Parameters
    ----------
    :param x: positive definite matrix

    Returns
    -------
    :return: logm(x)
    """
    eigendecomposition = torch.symeig(x, eigenvectors=True)

    eigenvectors = eigendecomposition.eigenvectors
    log_eigenvalues = torch.log(eigendecomposition.eigenvalues)  # Assume real eigenvalues (first column only)

    return torch.mm(eigenvectors, torch.mm(torch.diag(log_eigenvalues), torch.inverse(eigenvectors)))


def sqrtm_torch(x):
    """
    This function computes the square root of a matrix.

    Parameters
    ----------
    :param x: positive definite matrix

    Returns
    -------
    :return: sqrtm(x)
    """
    eigendecomposition = torch.symeig(x, eigenvectors=True)

    eigenvectors = eigendecomposition.eigenvectors
    sqrt_eigenvalues = torch.sqrt(eigendecomposition.eigenvalues)  # Assume real eigenvalues (first column only)

    return torch.mm(eigenvectors, torch.mm(torch.diag(sqrt_eigenvalues), torch.inverse(eigenvectors)))


def affine_invariant_distance_torch(x1, x2, diagonal_distance=False):
    """
    Compute the affine invariant distance between points on the SPD manifold

    Parameters
    ----------
    :param x1: set of SPD matrices      (N1 x d x d or b1 x ... x bk x N1 x d x d)
    :param x2: set of SPD matrices      (N2 x d x d or b1 x ... x bk x N1 x d x d)

    Optional parameters
    -------------------
    :param diagonal_distance: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold affine-invariante distance between points in x1 and x2
        (N1 x N2 or b1 x ... x bk x N1 x N2)
    """
    # If diag, x1 must be equal to x2 and we can return zeros.
    if diagonal_distance is True:
        shape = list(x2.shape)[:-2]
        shape.append(1)
        return torch.zeros(shape, dtype=x1.dtype)

    dim = x1.shape[-1]

    # Expand dimensions to compute all matrix-matrix distances
    x1 = x1.unsqueeze(-3)
    x2 = x2.unsqueeze(-4)

    # Method 1: compute x1^(-1)
    # x1_inv = torch.inverse(x1)  # method 1: uses inv(x1)x2

    # Method 2: uses the Cholesky decomposition to compute x1^(-0.5)
    x1_chol = torch.cholesky(x1)
    x1_chol_inv = torch.inverse(x1_chol)

    # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
    # x1_inv = torch.cat(x2.shape[-3] * [x1_inv], dim=-3) # method 1
    x1_chol_inv = torch.cat(x2.shape[-3] * [x1_chol_inv], dim=-3)
    x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

    # Compute the distance between each pair of matrices
    # Method 1: compute x1\x2
    # x1_inv_x2 = torch.bmm(x1_inv.view(-1, dim, dim), x2.view(-1, dim, dim))  # method 1
    # x1_inv_x2 = torch.solve(x1, x2).solution.view(-1, dim, dim)

    # Method 2: compute x1^(-0.5)*x2*x1^(-0.5)
    # The advantage of this method is that the resulting matrix is symmetric => we can use symeig for the eigenvalues.
    x1_inv_x2_x1_inv = torch.bmm(torch.bmm(x1_chol_inv.view(-1, dim, dim), x2.view(-1, dim, dim)),
                                 x1_chol_inv.view(-1, dim, dim).transpose(-2, -1))

    # x1_inv_x2_x1_inv += 1e-10 * torch.rand(x1_inv_x2_x1_inv.shape)

    # Compute norm(logm(x1\x2))
    eig_values = torch.zeros(x1_inv_x2_x1_inv.shape[0], dim)
    for i in range(x1_inv_x2_x1_inv.shape[0]):
        eig_values[i] = torch.symeig(x1_inv_x2_x1_inv[i], eigenvectors=True).eigenvalues  # Eigenvalue True necessary
        #  for derivation
    # Reshape
    shape = list(x2.shape)[:-2]
    shape.append(eig_values.shape[-1])
    eigv = eig_values.view(shape)

    logeigv = torch.log(eigv)
    logeigv2 = logeigv * logeigv
    sumlogeigv2 = torch.sum(logeigv2, dim=-1)
    return torch.sqrt(sumlogeigv2 + 1e-15).double()
    # return torch.sqrt(torch.sum(logeigv * logeigv, dim=-1) + 1e-15).double()


def frobenius_distance_torch(x1, x2, diagonal_distance=False):
    """
    Compute the Frobenius distance between matrix points

    Parameters
    ----------
    :param x1: set of matrices              (N1 x d1 x d2 or b1 x ... x bk x N1 x d1 x d2)
    :param x2: set of matrices              (N2 x d1 x d2 or b1 x ... x bk x N2 x d1 x d2)

    Optional parameters
    -------------------
    :param diagonal_distance: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of Frobenius distance between points in x1 and x2       (N1 x N2 or b1 x ... x bk x N1 x N2)
    """
    # If diag, x1 must be equal to x2 and we can return zeros.
    if diagonal_distance is True:
        shape = list(x2.shape)[:-2]
        shape.append(1)
        return torch.zeros(shape, dtype=x1.dtype)

    # Expand dimensions to compute all matrix-matrix distances
    x1 = x1.unsqueeze(-3)
    x2 = x2.unsqueeze(-4)

    # Repeat x and y data along 1- and 0- dimensions to have ndata_x x ndata_y x dim x dim arrays
    x1 = torch.cat(x2.shape[-3] * [x1], dim=-3)
    x2 = torch.cat(x1.shape[-4] * [x2], dim=-4)

    # Compute the distance between each pair of matrices
    return torch.norm(torch.add(x1, -x2) + 1e-15, dim=[-2, -1]).double()


def vector_to_symmetric_matrix_mandel_torch(vectors):
    """
    This function transforms vectors to symmetric matrices using Mandel notation

    Parameters
    ----------
    :param vectors: set of vectors          N x d_vec or b1 x ... x bk x N x d_vec

    Returns
    -------
    :return: set of symmetric matrices      N x d_mat x d_mat b1 x ... x bk x N x d_mat x d_mat
    """
    init_shape = list(vectors.shape)
    vectors = vectors.view(-1, init_shape[-1])
    n_mat = vectors.shape[0]
    d_vec = vectors.shape[1]
    d_mat = int((-1.0 + (1.0 + 8.0 * d_vec) ** 0.5) / 2.0)

    matrices = torch.zeros(n_mat, d_mat, d_mat, dtype=vectors.dtype)

    for n in range(n_mat):
        vector = vectors[n]
        matrix = torch.diag(vector[0:d_mat])

        id = np.cumsum(range(d_mat, 0, -1))

        for i in range(0, d_mat - 1):
            matrix += torch.diag(vector[range(id[i], id[i + 1])], i + 1) / 2.0 ** 0.5
            matrix += torch.diag(vector[range(id[i], id[i + 1])], -i - 1) / 2.0 ** 0.5

        matrices[n] = matrix

    new_shape = init_shape[:-1]
    new_shape.append(d_mat)
    new_shape.append(d_mat)
    return matrices.view(new_shape)


def symmetric_matrix_to_vector_mandel_torch(matrices):
    """
    This function transforms symmetric matrices to vectors using Mandel notation

    Parameters
    ----------
    :param matrices: set of symmetric matrices      N x d_mat x d_mat b1 x ... x bk x N x d_mat x d_mat

    Returns
    -------
    :return: set of vectors                         N x d_vec or b1 x ... x bk x N x d_vec
    """
    init_shape = list(matrices.shape)
    d_mat = matrices.shape[-1]
    matrices = matrices.view(-1, d_mat, d_mat)
    n_mat = matrices.shape[0]

    vectors = []
    for n in range(n_mat):
        vector = matrices[n].diag()
        for d in range(1, d_mat):
            # Consider both diagonals for gradient computation
            vector = torch.cat((vector, 0.5 * (2.0**0.5*matrices[n].diag(d) + 2.0**0.5*matrices[n].diag(-d))))
        vectors.append(vector[None])

    vectors = torch.cat(vectors)
    new_shape = init_shape[:-2]
    new_shape.append(vectors.shape[-1])

    return vectors.view(new_shape).type(matrices.dtype)
