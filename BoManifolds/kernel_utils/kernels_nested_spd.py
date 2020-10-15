import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Interval

import pymanopt.manifolds as pyman_man

from BoManifolds.Riemannian_utils.spd_utils_torch import vector_to_symmetric_matrix_mandel_torch, \
    affine_invariant_distance_torch, frobenius_distance_torch, logm_torch
from BoManifolds.nested_mappings.nested_spd_utils import projection_from_spd_to_nested_spd

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


class NestedSpdAffineInvariantGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between projected input points from a
    high-dimensional SPD manifold to a low-dimensional SPD manifold via nested-spheres projections using the
    affine-invariant distance.

    Attributes
    ----------
    self.beta_min: minimum value of the inverse square lengthscale parameter beta
    self.dim, dimension of the ambient high-dimensional sphere manifold
    self.latent_dim, dimension of the latent low-dimensional sphere manifold

    Properties
    ----------
    self.beta, inverse square lengthscale parameter beta
    self.projection_matrix, projection matrix of the nested SPD projection


    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

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
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(NestedSpdAffineInvariantGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min
        self.dim = dim
        self.latent_dim = latent_dim

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1,
                                                                                          beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee the positive-definiteness of the
        #  kernel.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))

        # Add projection parameters
        self.raw_projection_matrix_manifold = pyman_man.Grassmann(self.dim, self.latent_dim)
        self.register_parameter(name="raw_projection_matrix",
                                parameter=torch.nn.Parameter(torch.Tensor(self.raw_projection_matrix_manifold.rand()).
                                                             repeat(*self.batch_shape, 1, 1)))

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

    @property
    def projection_matrix(self):
        return self.raw_projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._set_projection_matrix(value)

    def _set_projection_matrix(self, value):
        self.initialize(raw_projection_matrix=value)

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to the ambient high-dim. SPD manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diagonal_distance: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Projection from the SPD manifold to the latent low-dimensional SPD manifold
        px1 = projection_from_spd_to_nested_spd(x1, self.projection_matrix)
        px2 = projection_from_spd_to_nested_spd(x2, self.projection_matrix)

        # Compute distance
        distance = affine_invariant_distance_torch(px1, px2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.mul(self.beta.double()))

        return exp_component


class NestedSpdLogEuclideanGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between projected input points from a
    high-dimensional SPD manifold to a low-dimensional SPD manifold via nested-spheres projections using the
    log-Euclidean distance.

     Attributes
    ----------
    self.dim, dimension of the ambient high-dimensional sphere manifold
    self.latent_dim, dimension of the latent low-dimensional sphere manifold

    Properties
    ----------
    self.projection_matrix, projection matrix of the nested SPD projection

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, dim, latent_dim, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(NestedSpdLogEuclideanGaussianKernel, self).__init__(ard_num_dims=None, **kwargs)
        self.dim = dim
        self.latent_dim = latent_dim

        # Add projection parameters
        self.raw_projection_matrix_manifold = pyman_man.Grassmann(self.dim, self.latent_dim)
        self.register_parameter(name="raw_projection_matrix",
                                parameter=torch.nn.Parameter(torch.Tensor(self.raw_projection_matrix_manifold.rand()).
                                                             repeat(*self.batch_shape, 1, 1)))

    @property
    def projection_matrix(self):
        return self.raw_projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._set_projection_matrix(value)

    def _set_projection_matrix(self, value):
        self.initialize(raw_projection_matrix=value)

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        -------------------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diag: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------------------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Projection from the SPD manifold to the latent low-dimensional SPD manifold
        px1 = projection_from_spd_to_nested_spd(x1, self.projection_matrix)
        px2 = projection_from_spd_to_nested_spd(x2, self.projection_matrix)

        # Compute the log of the matrices
        # Reshape px1 to N x d x d format
        init_shape = list(px1.shape)
        px1 = px1.view(-1, self.latent_dim, self.latent_dim)
        nb_data = px1.shape[0]
        # Log
        log_px1 = torch.zeros_like(px1)
        for n in range(nb_data):
            log_px1[n] = logm_torch(px1[n])
        # Reshape to initial format
        log_px1 = log_px1.view(init_shape)

        # Reshape px2 to N x d x d format
        init_shape = list(px2.shape)
        px2 = px2.view(-1, self.latent_dim, self.latent_dim)
        nb_data = px2.shape[0]
        # Log
        log_px2 = torch.zeros_like(px2)
        for n in range(nb_data):
            log_px2[n] = logm_torch(px2[n])
        # Reshape to initial format
        log_px2 = log_px2.view(init_shape)

        # Compute distance
        distance = frobenius_distance_torch(log_px1, log_px2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))

        return exp_component
