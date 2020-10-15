import torch
import gpytorch
from gpytorch.constraints import GreaterThan

from BoManifolds.Riemannian_utils.sphere_utils_torch import sphere_distance_torch

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


class SphereGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the sphere manifold.

    Attributes
    ----------
    self.beta_min, minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_the_sphere, point2_in_the_sphere, diagonal_matrix_flag=False, **params)

    Static methods
    --------------
    """
    def __init__(self, beta_min, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param beta_min: minimum value of the inverse square lengthscale parameter beta

        Optional parameters
        -------------------
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(SphereGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee positive-definiteness.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))

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

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Gaussian kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

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
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        distance2 = torch.mul(distance, distance)
        # Kernel
        exp_component = torch.exp(- distance2.mul(self.beta.double()))
        return exp_component


class SphereLaplaceKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Laplace covariance matrix between input points on the sphere manifold.
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SphereLaplaceKernel, self).__init__(has_lengthscale=True, ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """
        Computes the Laplace kernel matrix between inputs x1 and x2 belonging to a sphere manifold.

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
        :return: kernel matrix between x1 and x2
        """
        # Compute distance
        distance = sphere_distance_torch(x1, x2, diag=diag)
        # Kernel
        exp_component = torch.exp(- distance.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))
        return exp_component
