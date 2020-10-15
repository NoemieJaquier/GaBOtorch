import torch

from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch, rotation_from_sphere_points_torch

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


def projection_from_sphere_to_nested_sphere(x, sphere_axis, sphere_distance_to_axis):
    """
    This function computes the projection of data on a sphere Sd to a small circle in Sd (nested sphere).

    Parameters
    ----------
    :param x: data on the sphere Sd     [N x d] or [b1 x ... x bk x N x d]
    :param sphere_axis: axis of the nested sphere belonging to Sd
    :param sphere_distance_to_axis: distance from the axis w.r.t each point of the nested sphere

    Returns
    -------
    :return: data projected on the nested sphere    [N x d] or [b1 x ... x bk x N x d]
    """
    dim = x.shape[-1]

    # If x has more than two dimensions, fuse the first dimensions so that x = N x d
    init_shape = list(x.shape)
    x = x.view(-1, init_shape[-1])

    # Ensure that sphere axis is 2-dimensional
    if sphere_axis.dim() == 1:
        sphere_axis = sphere_axis.unsqueeze(-2)

    # Here, we first rotate the data, so that the axis is aligned with the north pole. For some reason,
    # this seems to limit the numerical errors, even more when the distance between the axis and the data is very small.
    # The data are finally rotated back once projected into the nested sphere.
    # Rotation matrix to rotate the axis to the north pole
    north_pole = torch.zeros_like(sphere_axis)
    north_pole[:, -1] = 1.
    rotation_matrix = rotation_from_sphere_points_torch(sphere_axis, north_pole)

    # Rotate data
    x_rotated = torch.mm(rotation_matrix, x.T).T

    # Compute the distance between the data and the subsphere axis
    distance_to_axis = sphere_distance_torch(x_rotated, north_pole)
    distance_to_axis = distance_to_axis.repeat((1, dim))

    # Project data to the nested sphere
    x_nested_sphere_rotated = torch.sin(sphere_distance_to_axis) * x_rotated \
                                + torch.sin(distance_to_axis - sphere_distance_to_axis) * north_pole
    x_nested_sphere_rotated_rescaled = x_nested_sphere_rotated / (torch.sin(distance_to_axis)
                                                                  + 1e-6*torch.ones_like(distance_to_axis))

    # Rotate the data back
    x_nested_sphere = torch.mm(rotation_matrix.T, x_nested_sphere_rotated_rescaled.T).T

    # Back to initial data structure
    new_shape = init_shape[:-1]
    new_shape.append(x_nested_sphere.shape[-1])

    return x_nested_sphere.view(new_shape)


def projection_from_sphere_to_next_subsphere(x, sphere_axis, sphere_distance_to_axis):
    """
    This function computes the projection of data on a sphere Sd to a subsphere Sd-1.
    The data are first projected to a nested sphere, which is then identified with Sd-1.

    Parameters
    ----------
    :param x: data on the sphere Sd     [N x d] or [b1 x ... x bk x N x d]
    :param sphere_axis: axis of the nested sphere belonging to Sd
    :param sphere_distance_to_axis: distance from the axis w.r.t each point of the nested sphere

    Returns
    -------
    :return: data projected on the subsphere    [N x d-1] or [b1 x ... x bk x N x d]
    """
    # If x has more than two dimensions, fuse the first dimensions so that x = N x d
    init_shape = list(x.shape)
    x = x.view(-1, init_shape[-1])

    # Ensure that sphere axis is 2-dimensional
    if sphere_axis.dim() == 1:
        sphere_axis = sphere_axis.unsqueeze(-2)

    # Define the north pole
    north_pole = torch.zeros_like(sphere_axis)
    north_pole[:, -1] = 1.

    # Projection onto the nested sphere defined by the axis and distance
    x_nested_sphere = projection_from_sphere_to_nested_sphere(x, sphere_axis, sphere_distance_to_axis)

    # Rotation matrix to rotate the axis to the north pole
    rotation_matrix = rotation_from_sphere_points_torch(sphere_axis, north_pole)

    # Identification of the nested sphere with the subsphere of radius 1
    x_subsphere = torch.mm(rotation_matrix[:-1, :], x_nested_sphere.T).T / (torch.sin(sphere_distance_to_axis) +
                                                                            1e-6 *
                                                                            torch.ones_like(sphere_distance_to_axis))

    # For numerical reason, we ensure here that the norm is exactly 1.
    norm_x_subsphere = torch.norm(x_subsphere, dim=[-1]).unsqueeze(-1).repeat(1, x_subsphere.shape[-1])
    x_subsphere = x_subsphere / (norm_x_subsphere + 1e-6 * torch.ones_like(norm_x_subsphere))

    # Back to initial data structure
    new_shape = init_shape[:-1]
    new_shape.append(x_subsphere.shape[-1])

    return x_subsphere.view(new_shape)


def projection_from_sphere_to_subsphere(x, sphere_axes, sphere_distances_to_axes):
    """
    This function computes the projection of data on a sphere Sd to a subsphere Sd-r.
    For each dimension, the data are first projected to a nested sphere Si, which is then identified with Si-1.

    Parameters
    ----------
    :param x: data on the sphere Sd     [N x d]
    :param sphere_axes: axes of the nested spheres belonging to [Sd, Sd-1, ..., Sd-r+1]
    :param sphere_distances_to_axes: distances from the axes w.r.t each point of the nested spheres of
    [Sd, Sd-1, ..., Sd-r+1]

    Returns
    -------
    :return: data projected on the subspheres of dimension d-1 to d-r    list([N x d-1], ..., [N x d-r])
    """
    x_subsphere = [x]

    # Transform parameters into lists
    if not isinstance(sphere_axes, list):
        sphere_axes = [sphere_axes]
    if not isinstance(sphere_distances_to_axes, list):
        sphere_distances_to_axes = [sphere_distances_to_axes]

    # Compute the subsphere for each dimension from the precedent subsphere
    for s in range(len(sphere_axes)):
        x_subsphere.append(projection_from_sphere_to_next_subsphere(x_subsphere[-1], sphere_axes[s],
                                                                    sphere_distances_to_axes[s]))

    return x_subsphere


def projection_from_subsphere_to_next_sphere(x_subsphere, sphere_axis, sphere_distance_to_axis):
    """
    This function computes the projection of data from a subsphere Sd-1 to a sphere Sd.
    The data are first identified on a nested sphere with the axis at the north pole, and then rotated according to the
    nested sphere axis.

    Parameters
    ----------
    :param x_subsphere: data on the subsphere Sd-1     [N x d-1]
    :param sphere_axis: axis of the nested sphere belonging to Sd
    :param sphere_distance_to_axis: distance from the axis w.r.t each point of the nested sphere

    Returns
    -------
    :return: data projected on the sphere of dimension d    [N x d]
    """
    if sphere_axis.dim() == 1:
        sphere_axis = sphere_axis.unsqueeze(-2)

    # Define the north pole
    north_pole = torch.zeros_like(sphere_axis)
    north_pole[:,  -1] = 1.

    # Rotation matrix to rotate the north pole to the axis
    rotation_matrix = rotation_from_sphere_points_torch(north_pole, sphere_axis)

    # Projection of the data from the subsphere to the sphere
    cos_vector = torch.cos(sphere_distance_to_axis) * torch.ones(x_subsphere.shape[0], 1, dtype=x_subsphere.dtype)
    x = torch.mm(rotation_matrix, torch.cat((torch.sin(sphere_distance_to_axis) * x_subsphere, cos_vector), 1).T).T

    return x


def projection_from_subsphere_to_sphere(x_subsphere, sphere_axes, sphere_distances_to_axes):
    """
    This function computes the projection of data from a subsphere Sd-r to a sphere Sd.
    For each dimension, the data are first identified on a nested sphere with the axis at the north pole, and then
    rotated according to the nested sphere axis.

    Parameters
    ----------
    :param x_subsphere: data on the subsphere Sd-1     [N x d-1]
    :param sphere_axes: axes of the nested spheres belonging to [Sd, Sd-1, ..., Sd-r+1]
    :param sphere_distances_to_axes: distances from the axes w.r.t each point of the nested spheres of
    [Sd, Sd-1, ..., Sd-r+1]

    Returns
    -------
    :return: data projected on the spheres of dimension d-r+1 to d    list([N x d-r+1], ..., [N x d])
    """
    x = [x_subsphere]

    # Transform parameters into lists
    if not isinstance(sphere_axes, list):
        sphere_axes = [sphere_axes]
    if not isinstance(sphere_distances_to_axes, list):
        sphere_distances_to_axes = [sphere_distances_to_axes]

    # Compute the sphere for each dimension from the precedent sphere
    nb_spheres = len(sphere_axes)
    for s in range(nb_spheres):
        x.append(projection_from_subsphere_to_next_sphere(x[-1], sphere_axes[nb_spheres-s-1],
                                                          sphere_distances_to_axes[nb_spheres-s-1]))

    return x



