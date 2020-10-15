import numpy as np
import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Interval

import pymanopt.manifolds as pyman_man

from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch
from BoManifolds.nested_mappings.nested_spheres_utils import projection_from_sphere_to_subsphere

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


class NestedSphereGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between projected input points from a
    high-dimensional sphere manifold to a low-dimensional sphere manifold via nested-spheres projections.

    Attributes
    ----------
    self.beta_min, minimum value of the inverse square lengthscale parameter beta
    self.dim, dimension of the ambient high-dimensional sphere manifold
    self.latent_dim, dimension of the latent low-dimensional sphere manifold

    Properties
    ----------
    self.beta, inverse square lengthscale parameter beta
    self.sphere_axes, axes of the nested spheres belonging to [Sd, Sd-1, ..., Sd-r+1]
    self.sphere_distances_to_axes, distances from the axes w.r.t each point of the nested spheres of [Sd, Sd-1, ...,
        Sd-r+1]

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, dim, latent_dim, beta_min, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the ambient high-dimensional sphere manifold
        :param latent_dim: dimension of the latent low-dimensional sphere manifold
        :param beta_min: minimum value of the inverse square lengthscale parameter beta

        Optional parameters
        -------------------
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(NestedSphereGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min
        self.dim = dim
        self.latent_dim = latent_dim

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee positive-definiteness.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))

        # Add projection parameters
        for d in range(self.dim, self.latent_dim, -1):
            # Axes parameters
            # Register
            axis_name = "raw_axis_S" + str(d)
            # axis = torch.zeros(1, d)
            # axis[:, 0] = 1
            axis = torch.randn(1, d)
            axis = axis / torch.norm(axis)
            axis = axis.repeat(*self.batch_shape, 1, 1)
            self.register_parameter(name=axis_name,
                                    parameter=torch.nn.Parameter(axis))
            # Corresponding manifold
            axis_manifold_name = "raw_axis_S" + str(d) + "_manifold"
            setattr(self, axis_manifold_name, pyman_man.Sphere(d))

        # Distance to axis (constant), fixed at pi/2
        self.distances_to_axis = [np.pi/2 *torch.ones(1, 1) for d in range(self.dim, self.latent_dim, -1)]

    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_beta(value)

    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))
        # self.initialize(**{'raw_beta': self.raw_beta_constraint.inverse_transform(value)})

    @property
    def axes(self):
        return [self._parameters["raw_axis_S" + str(d)] for d in range(self.dim, self.latent_dim, -1)]

    @axes.setter
    def axes(self, values_list):
        self._set_axes(values_list)

    def _set_axes(self, values_list):
        for d in range(self.dim, self.latent_dim, -1):
            value = values_list[self.dim-d]
            axis_name = "raw_axis_S" + str(d)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self._parameters[axis_name])
            self.initialize(**{axis_name: value})

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to the ambient high-dim. sphere manifold.

        Parameters
        ----------
        :param x1: input points on the sphere
        :param x2: input points on the sphere

        Optional parameters
        -------------------
        :param diag: Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between p(x1) and p(x2)
        """
        # Projection from the sphere to the latent low-dimensional sphere
        px1 = projection_from_sphere_to_subsphere(x1, self.axes, self.distances_to_axis)[-1]
        px2 = projection_from_sphere_to_subsphere(x2, self.axes, self.distances_to_axis)[-1]

        # Compute distance
        distance = sphere_distance_torch(px1, px2, diag=diag)
        distance2 = torch.mul(distance, distance)
        # Kernel
        exp_component = torch.exp(- distance2.mul(self.beta))
        return exp_component
