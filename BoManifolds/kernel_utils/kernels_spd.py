import torch
import gpytorch
from gpytorch.constraints import GreaterThan

from BoManifolds.Riemannian_utils.spd_utils_torch import logm_torch, vector_to_symmetric_matrix_mandel_torch, \
    affine_invariant_distance_torch, frobenius_distance_torch


'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''


class SpdAffineInvariantGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold using
    the affine-invariant distance.

    Attributes
    ----------
    self.beta_min: minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, beta_min, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param beta_min: minimum value of the inverse square lengthscale parameter beta
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(SpdAffineInvariantGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1,
                                                                                          beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee the positive-definiteness of
        # the kernel.
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

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

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

        # Compute distance
        distance = affine_invariant_distance_torch(x1, x2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.mul(self.beta.double()))

        return exp_component


class SpdAffineInvariantLaplaceKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Laplace covariance matrix between input points on the SPD manifold.

    Attributes
    ----------
    self.beta_min: minimum value of the inverse square lengthscale parameter beta

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

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
        super(SpdAffineInvariantLaplaceKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1,
                                                                                          beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda: self.beta, lambda v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee the positive-definiteness of
        # the kernel.
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

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Laplace kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        ----------
        :param x1: input points on the SPD manifold
        :param x2: input points on the SPD manifold

        Optional parameters
        -------------------
        :param diagonal_distance: Whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`
        :param params: additional parameters

        Returns
        -------
        :return: kernel matrix between x1 and x2
        """
        # Transform input vector to matrix
        x1 = vector_to_symmetric_matrix_mandel_torch(x1)
        x2 = vector_to_symmetric_matrix_mandel_torch(x2)

        # Compute distance
        distance = affine_invariant_distance_torch(x1, x2, diagonal_distance=diagonal_distance)

        exp_component = torch.exp(- distance.mul(self.beta.double()))

        return exp_component


class SpdFrobeniusGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between symmetric matrix input points using
    the Frobenius distance.

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, **kwargs):
        """
        Initialisation

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdFrobeniusGaussianKernel, self).__init__(ard_num_dims=None, **kwargs)

    def forward(self, x1, x2, diagonal_distance=False, **params):
        """
        Compute the Gaussian kernel matrix between inputs x1 and x2 belonging to a SPD manifold.

        Parameters
        -------------------
        :param x1: symmetric input points
        :param x2: symmetric input points

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

        # Compute distance
        distance = frobenius_distance_torch(x1, x2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))

        return exp_component


class SpdLogEuclideanGaussianKernel(gpytorch.kernels.Kernel):
    """
    Instances of this class represent a Gaussian (RBF) covariance matrix between input points on the SPD manifold using
    the log-Euclidean distance.

    Methods
    -------
    forward(point1_in_SPD, point2_in_SPD, diagonal_matrix_flag=False, **params):

    Static methods
    --------------
    """
    def __init__(self, **kwargs):
        """
        Initialisation.

        Optional parameters
        -------------------
        :param kwargs: additional arguments
        """
        self.has_lengthscale = True
        super(SpdLogEuclideanGaussianKernel, self).__init__(ard_num_dims=None, **kwargs)

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

        # Compute the log of the matrices
        init_shape = list(x1.shape)
        dim = x1.shape[-1]
        x1 = x1.view(-1, dim, dim)
        nb_data = x1.shape[0]
        log_x1 = torch.zeros_like(x1)
        for n in range(nb_data):
            log_x1[n] = logm_torch(x1[n])
        log_x1 = log_x1.view(init_shape)

        init_shape = list(x2.shape)
        x2 = x2.view(-1, dim, dim)
        nb_data = x2.shape[0]
        log_x2 = torch.zeros_like(x2)
        for n in range(nb_data):
            log_x2[n] = logm_torch(x2[n])
        log_x2 = log_x2.view(init_shape)

        # Compute distance
        distance = frobenius_distance_torch(log_x1, log_x2, diagonal_distance=diagonal_distance)
        distance2 = torch.mul(distance, distance)

        exp_component = torch.exp(- distance2.div(torch.mul(self.lengthscale.double(), self.lengthscale.double())))

        return exp_component


