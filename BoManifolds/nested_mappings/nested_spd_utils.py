import torch

from BoManifolds.Riemannian_utils.spd_utils_torch import sqrtm_torch

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def projection_from_spd_to_nested_spd(x_spd, projection_matrix):
    """
    This function projects data from the SPD manifold S^D_++ to a lower dimensional SPD manifold S^d_++.
    This is done by multiplying each DxD SPD data on the left and on the right by an orthogonal matrix belonging to the
    Grassmann manifold G(D,d). The resulting SPD data of dimension dxd with d < D are given by Y = W'XW,
    with X \in S^D_++ and W \in G(D,d).
    This projection was proposed in  ["Dimensionality Reduction on SPD Manifolds: The Emergence of Geometry-Aware
    Methods", M. Harandi, S. Salzmann and R. Harley. PAMI 2018].

    Parameters
    ----------
    :param x_spd: SPD matrix or set of SPD matrices                                 (D x D or b1 x ... x b_k x D x D)
    :param projection_matrix: element of the Grassmann manifold                     (D x d)

    Returns
    -------
    :return: low dimensional SPD matrix or set of low dimensional SPD matrices      (d x d or b1 x ... x b_k x d x d)
    """
    low_dimension = projection_matrix.shape[1]

    # Reshape x_spd to N x d x d format
    init_shape = list(x_spd.shape)
    dimension = x_spd.shape[-1]
    x_spd = x_spd.view(-1, dimension, dimension)
    nb_data = x_spd.shape[0]

    # Augment projection matrix
    projection_matrix = projection_matrix.unsqueeze(0)
    projection_matrix = projection_matrix.repeat([nb_data, 1, 1])

    # Project data to SPD matrix of lower dimension
    x_spd_low_dimension = torch.bmm(torch.bmm(projection_matrix.transpose(-2, -1), x_spd), projection_matrix)

    # Back to initial shape
    new_shape = init_shape[:-2] + [low_dimension, low_dimension]
    return x_spd_low_dimension.view(new_shape)


def projection_from_nested_spd_to_spd(x_spd_low_dimension, projection_matrix, projection_complement_matrix,
                                      bottom_spd_matrix, contraction_matrix):
    """
    This function is an approximation of the inverse of the function projection_from_spd_to_nested_spd.
    It maps low-dimensional SPD matrices to the original SPD space.
    To do so, we consider that the nested SPD matrix Y = W'XW is the d x d upper-left part of the rotated matrix
    Xr = R'XR, where R = [W, V] and Xr = [Y B; B' C].
    In order to recover X, we assume a constant SPD matrix C, and B = Y^0.5*K*C^0.5 to ensure the PDness of Xr, with
    K a contraction matrix (norm(K) <=1). We first reconstruct Xr, and then X as X = RXrR'.
    Note that W and V belong to Grassmann manifolds, W \in G(D,d) and V \in G(D,D-d), and must have orthonormal columns,
     so that W'V = 0.

    Parameters
    ----------
    :param x_spd_low_dimension: low dimensional SPD matrix or set of low dimensional SPD matrices   (d x d or N x d x d)
    :param projection_matrix: element of the Grassmann manifold                                     (D x d)
    :param projection_complement_matrix: element of the Grassmann manifold                          (D x D-d)
        Note that we must have torch.mm(projection_complement_matrix.T, projection_matrix) = 0.
    :param bottom_spd_matrix: bottom-right part of the rotated SPD matrix                           (D-d, D-d)
    :param contraction_matrix: matrix whose norm is <=1                                             (d x D-d)

    Returns
    -------
    :return: SPD matrix or set of SPD matrices                                                      (D x D or N x D x D)
    """

    # Type
    torch_type = x_spd_low_dimension.dtype

    # Number of data
    if x_spd_low_dimension.ndim == 2:
        nb_data = 1
        x_spd_low_dimension = torch.unsqueeze(x_spd_low_dimension, 0)
        one_data_output = True  # To return a 2D SPD matrix
    else:
        nb_data = x_spd_low_dimension.shape[0]
        one_data_output = False  # To return a 3D array of nb_data SPD matrices

    # SPD matrices array initialization
    dimension = projection_matrix.shape[0]
    x_spd = torch.zeros((nb_data, dimension, dimension), dtype=torch_type)

    # Compute rotation matrix
    rotation_matrix = torch.cat((projection_matrix, projection_complement_matrix), dim=1)
    # inverse_rotation_matrix = torch.inverse(rotation_matrix)

    # Compute sqrtm of the bottom block
    sqrt_bottom_spd_matrix = sqrtm_torch(bottom_spd_matrix)

    # Solve the equation for each data
    for n in range(nb_data):
        # Compute sqrtm of the top block
        sqrt_top_spd_matrix = sqrtm_torch(x_spd_low_dimension[n])

        # Side block
        side_block = torch.mm(torch.mm(sqrt_top_spd_matrix, contraction_matrix), sqrt_bottom_spd_matrix)

        # Reconstruct full SPD matrix
        x_spd_reconstructed = torch.cat((torch.cat((x_spd_low_dimension[n], side_block), dim=1),
                                         torch.cat((side_block.T, bottom_spd_matrix), dim=1)), dim=0)

        # Rotate the matrix back to finalize the reconstruction
        x_spd[n] = torch.mm(rotation_matrix, torch.mm(x_spd_reconstructed, rotation_matrix.T))

    if one_data_output:
        x_spd = x_spd[0]

    return x_spd
