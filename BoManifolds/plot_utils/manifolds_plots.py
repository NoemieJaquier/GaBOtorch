import numpy as np
import scipy as sc

from BoManifolds.Riemannian_utils.utils import rotation_matrix_from_axis_angle
from BoManifolds.Riemannian_utils.sphere_utils import get_axisangle
import BoManifolds.Riemannian_utils.sphere_utils as sphere

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def plot_sphere(ax, base=None, color=None, alpha=0.8, r=0.99, linewidth=0, lim=1.1, n_elems=100, **kwargs):
    """
    Plots a sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes

    Optional parameters
    -------------------
    :param color: color of the surface
    :param alpha: transparency index
    :param r: radius
    :param linewidth: linewidth of sphere lines
    :param lim: axes limits
    :param n_elems: number of points in the surface
    :param kwargs:

    Returns
    -------
    :return: -
    """
    if base is None:
        base = [0, 0, 1]
    else:
        if len(base) != 3:
            base = [0, 0, 1]
            print('Base was set to its default value as a wrong argument was given!')

    if color is None:
        color = [0.8, 0.8, 0.8]
    else:
        if len(color) != 3:
            color = [0.8, 0.8, 0.8]
            print('Sphere color was set to its default value as a wrong color argument was given!')

    u = np.linspace(0, 2 * np.pi, n_elems)
    v = np.linspace(0, np.pi, n_elems)

    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha, **kwargs)
    # ax.plot(xs=[base[0]], ys=[base[1]], zs=[base[2]], marker='*', color=color)

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])


def plot_sphere_tangent_plane(ax, base, l_vert=1, color='w', alpha=0.15, linewidth=0.5, **kwargs):
    """
    Plots tangent plane of a point lying on the sphere manifold
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes
    :param base: base point of the tangent space

    Optional parameters
    -------------------
    :param l_vert: length/width of the displayed plane
    :param color: color of the plane
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
    (axis, ang) = get_axisangle(base)
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
    points = (T.dot(X).T + base).T
    psurf = points.reshape((-1, 2, 2))

    ax.plot_surface(psurf[0, :], psurf[1, :], psurf[2, :], color=color, alpha=alpha, linewidth=0, **kwargs)

    # Plot contours of the tangent space
    points_lines = points[:, [0, 1, 3, 2, 0]]
    ax.plot(points_lines[0], points_lines[1], points_lines[2], color='black', linewidth=linewidth)


def plot_gaussian_on_sphere(ax, mu, sigma, color='red', linewidth=2, linealpha=1, **kwargs):
    """
    Plots (mean and) covariance in the sphere manifold.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: figure axes
    :param mu: mean (point on the manifold)
    :param sigma: covariance belonging to the tangent space of the mean

    Optional parameters
    -------------------
    :param color: color of the Gaussian
    :param linewidth: linewidth of the covariance
    :param linealpha: transparency index for the lines
    :param planealpha: transparency index for planes
    :param label:
    :param showtangent:
    :param kwargs:

    Returns
    -------
    :return: -
    """

    # Plot Gaussian
    # - Generate Points @ Identity:
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    R = np.real(sc.linalg.sqrtm(1.0 * sigma))
    # Rotation for covariance
    # (axis, angle) = get_axisangle(mu)
    # R = R_from_axis_angle(axis, angle).dot(R)  # Rotation for manifold location

    points = np.vstack((np.cos(t), np.sin(t), np.ones(nbDrawingSeg)))

    if np.ndim(mu) < 2:
        mu = mu[:, None]
    # points = R.dot(points) + mu
    points2 = R.dot(points) + mu
    points = sphere.expmap(R.dot(points), mu)

    # l, = ax.plot(xs=mu[0, None], ys=mu[1, None], zs=mu[2, None], marker='.', color=color, alpha=linealpha,
    #              label=label, **kwargs)  # Mean

    ax.plot(xs=points[0, :], ys=points[1, :], zs=points[2, :],
            color=color,
            linewidth=linewidth,
            markersize=2, alpha=linealpha, **kwargs)  # Contour


def plot_spd_cone(ax, r=1., color=[0.8, 0.8, 0.8], n_elems=50, linewidth=2., linewidth_axes=1., alpha=0.3, lim_fact=0.6,
                  l1=47, l2=30):
    """
    Plot the 2x2 SPD cone

    Parameters
    ----------
    :param ax: figure acis
    :param r: radius of the cone
    :param color: color of the surface of the cone
    :param n_elems: number of elements used to plot the cone
    :param linewidth: linewidth of the borders of the cone
    :param linewidth_axes: linewidth of the axis of the symmetric space (plotted at the origin)
    :param alpha: transparency factor
    :param lim_fact: factor for the axis length
    :param l1: index of the first line plotted to represent the border of the cone
    :param l2: index of the second line plotted to represent the border of the cone

    Returns
    -------
    :return: -
    """

    phi = np.linspace(0, 2 * np.pi, n_elems)

    # Rotation of 45° of the cone
    dir = np.cross(np.array([1, 0, 0]), np.array([1., 1., 0.]))
    R = rotation_matrix_from_axis_angle(dir, np.pi / 4.)

    # Points of the cone
    xyz = np.vstack((r * np.ones(n_elems), r * np.sin(phi), r / np.sqrt(2) * np.cos(phi)))

    xyz = R.dot(xyz)

    x = np.vstack((np.zeros(n_elems), xyz[0]))
    y = np.vstack((np.zeros(n_elems), xyz[1]))
    z = np.vstack((np.zeros(n_elems), xyz[2]))

    # Draw cone
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color=color, linewidth=linewidth, alpha=alpha)

    ax.plot(xyz[0], xyz[1], xyz[2], color='k', linewidth=linewidth)
    ax.plot(x[:, l1], y[:, l1], z[:, l1], color='k', linewidth=linewidth)
    ax.plot(x[:, l2], y[:, l2], z[:, l2], color='k', linewidth=linewidth)

    # Draw axis
    lim = lim_fact * r
    x_axis = np.array([[0, lim / 2], [0, 0], [0, 0]])
    y_axis = np.array([[0, 0], [0, lim / 2], [0, 0]])
    z_axis = np.array([[0, 0], [0, 0], [0, lim / 2]])

    ax.plot(x_axis[0], x_axis[1], x_axis[2], color='k', linewidth=linewidth_axes)
    ax.plot(y_axis[0], y_axis[1], y_axis[2], color='k', linewidth=linewidth_axes)
    ax.plot(z_axis[0], z_axis[1], z_axis[2], color='k', linewidth=linewidth_axes)

    # Set limits
    ax.set_xlim([-lim/2, 3.*lim/2])
    ax.set_ylim([-lim/2, 3*lim/2])
    ax.set_zlim([-lim, lim])


