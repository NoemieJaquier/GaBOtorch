import torch
'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def post_processing_init_sphere_torch(x):
    """
    This function post-processes vectors, so that its norm is equal to 1.

    Parameters
    ----------
    :param x: d-dimensional vectors [N x d]

    Returns
    -------
    :return: unit-norm vectors [N x d]

    """
    return x / torch.cat(x.shape[-1] * [torch.norm(x, dim=[-1]).unsqueeze(-1)], dim=-1)
