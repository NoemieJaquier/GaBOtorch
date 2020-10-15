import numpy as np

from BoManifolds.Riemannian_utils.utils import rotation_matrix_from_axis_angle
from BoManifolds.Riemannian_utils.sphere_utils import get_axisangle

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def plot_ellipse3d(ax, ellipse_cov, center=None, color=None, alpha=0.2, linewidth=0, n_elems=50, **kwargs):
    """
    Plot a 3D ellipsoid
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes
    :param ellipse_cov: covariance matrix of the ellipsoid

    Optional parameters
    -------------------
    :param center: center of the ellipsoid
    :param color: color of the surface
    :param alpha: transparency index
    :param linewidth: linewidth of the surface
    :param n_elems: number of points in the surface
    :param kwargs:

    Returns
    -------
    :return: -
    """
    center = center or [0, 0, 0]
    color = color or [0.8, 0.8, 0.8]

    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    D, V = np.linalg.eig(ellipse_cov)
    D = np.real(D)
    V = np.real(V)

    x0 = D[0] * np.outer(np.cos(u), np.sin(v))
    y0 = D[1] * np.outer(np.sin(u), np.sin(v))
    z0 = D[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    xyz0 = np.stack((x0, y0, z0), axis=2)
    xyz0 = np.reshape(xyz0, (n_elems*n_elems, 3)).T
    xyz = np.dot(V, xyz0)
    xyz = np.reshape(xyz.T, (n_elems, n_elems, 3))

    x = xyz[:, :, 0] + center[0]
    y = xyz[:, :, 1] + center[1]
    z = xyz[:, :, 2] + center[2]

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, **kwargs)


def plot_plane(ax, normal_vector, point_on_plane, l_vert=1, color='w', line_color='black', alpha=0.15, linewidth=0.5,
               **kwargs):
    """
    Plot a plane in R3.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes
    :param normal_vector: normal vector of the plane
    :param point_on_plane: point lying on the plane

    Optional parameters
    -------------------
    :param l_vert: length/width of the displayed plane
    :param color: color of the plane
    :param line_color: color of the contour of the plane
    :param alpha: transparency index
    :param linewidth: linewidth of the border of the plane
    :param kwargs:

    Returns
    -------
    :return: -
    """
    # Tangent axis at 0 rotation:
    T0 = np.array([[1, 0], [0, 1], [0, 0]])

    # Rotation matrix with respect to zero:
    (axis, ang) = get_axisangle(normal_vector)
    R = rotation_matrix_from_axis_angle(axis, -ang)

    # Tangent axis in new plane:
    T = R.T.dot(T0)

    # Compute vertices of tangent plane at g
    hl = 0.5 * l_vert
    X = [[hl, hl],  # p0
         [hl, -hl],  # p1
         [-hl, hl],  # p2
         [-hl, -hl]]  # p3
    X = np.array(X).T
    points = (T.dot(X).T + point_on_plane).T
    psurf = points.reshape((-1, 2, 2))

    ax.plot_surface(psurf[0, :], psurf[1, :], psurf[2, :], color=color, alpha=alpha, linewidth=0, **kwargs)

    # Plot contours of the tangent space
    points_lines = points[:, [0, 1, 3, 2, 0]]
    ax.plot(points_lines[0], points_lines[1], points_lines[2], color=line_color, linewidth=linewidth)
