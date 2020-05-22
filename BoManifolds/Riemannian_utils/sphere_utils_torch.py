import torch
'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def sphere_distance_torch(x1, x2, diag=False):
    """
    This function computes the Riemannian distance between points on a sphere manifold.

    Parameters
    ----------
    :param x1: points on the sphere                                             N1 x dim or b1 x ... x bk x N1 x dim
    :param x2: points on the sphere                                             N2 x dim or b1 x ... x bk x N2 x dim

    Optional parameters
    -------------------
    :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.

    Returns
    -------
    :return: matrix of manifold distance between the points in x1 and x2         N1 x N2 or b1 x ... x bk x N1 x N2
    """
    if diag is False:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)

        # Repeat x and y data along -2 and -3 dimensions to have b1 x ... x ndata_x x ndata_y x dim arrays
        x1 = torch.cat(x2.shape[-2] * [x1], dim=-2)
        x2 = torch.cat(x1.shape[-3] * [x2], dim=-3)

        # Expand dimension to perform inner product
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-1)

        # Compute the inner product (should be [-1,1])
        inner_product = torch.bmm(x1.view(-1, 1, x1.shape[-1]), x2.view(-1, x2.shape[-2], 1)).view(x1.shape[:-2])

    else:
        # Expand dimensions to compute all vector-vector distances
        x1 = x1.unsqueeze(-1).transpose(-1, -2)
        x2 = x2.unsqueeze(-1)
        inner_product = torch.bmm(x1, x2).squeeze(-1)

    # Clamp in case any value is not in the interval [-1,1]
    # A small number is added/substracted to the bounds to avoid NaNs during backward computation.
    inner_product = inner_product.clamp(-1.+1e-15, 1.-1e-15)

    return torch.acos(inner_product)


def rotation_from_sphere_points_torch(x, y):
    """
    Gets the rotation matrix that moves x to y in the geodesic path on the sphere.
    Based on the equations of "Analysis of principal nested spheres", Sung et al. 2012 (appendix)

    Parameters
    ----------
    :param x: point on a sphere
    :param y: point on a sphere

    Returns
    -------
    :return: rotation matrix
    """
    if x.dim() is 1:
        x = x.unsqueeze(-2)
    if y.dim() is 1:
        y = y.unsqueeze(-2)

    dim = x.shape[1]

    # Compute the inner product
    inner_product = torch.mm(x, y.T)
    # Clamp in case any value is not in the interval [-1,1]
    # A small number is added/substracted to the bounds to avoid NaNs during backward computation.
    inner_product = inner_product.clamp(-1. + 1e-15, 1. - 1e-15)

    # Compute intermediate vector
    c_vec = x - y * inner_product
    c_vec = c_vec / torch.norm(c_vec)

    R = torch.eye(dim, dim, dtype=inner_product.dtype) + \
        torch.sin(torch.acos(inner_product)) * (torch.mm(y.T, c_vec) - torch.mm(c_vec.T, y)) + \
        (inner_product - 1.) * (torch.mm(y.T, y) + torch.mm(c_vec.T, c_vec))

    return R
