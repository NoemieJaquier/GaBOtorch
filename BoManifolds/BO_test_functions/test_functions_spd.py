import numpy as np
import torch

from BoManifolds.Riemannian_utils.spd_utils import vector_to_symmetric_matrix_mandel, symmetric_matrix_to_vector_mandel

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def ackley_function_spd(x, spd_manifold):
    """
    This function computes the Ackley function on the SPD manifold.
    The Ackley function is defined on the tangent space of a base point and projected to the manifold via the
    exponential map. The value of the function is therefore computed by projecting the point on the manifold to
    the tangent space of the base point and by computing the value of the Ackley function in this Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array. The input is
    given in Mandel notation.

    Note: the based point is obtained from and can be modified in the function "get_ackley_base".

    Parameters
    ----------
    :param x: point on the SPD manifold (in Mandel notation)    (torch tensor)
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Returns
    -------
    :return: value of the Ackley function at x                  (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Mandel notation dimension
    vector_dimension = int(dimension + dimension * (dimension - 1) / 2)

    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()

    if np.ndim(x) < 2:
        x = x[None]

    # To vector (Mandel notation)
    x = vector_to_symmetric_matrix_mandel(x[0])

    # Projection in tangent space of the base
    base = get_ackley_base(spd_manifold)
    x_proj = spd_manifold.log(base, x)

    # Vectorize to use only once the symmetric elements
    # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
    x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
    x_proj_vec[dimension:] /= 2. ** 0.5

    # Ackley function parameters
    a = 20
    b = 0.2
    c = 2*np.pi

    # Ackley function
    aexp_term = -a * np.exp(-b * np.sqrt(np.sum(x_proj_vec**2) / vector_dimension))
    expcos_term = - np.exp( np.sum(np.cos(c*x_proj_vec) / vector_dimension))
    y = aexp_term + expcos_term + a + np.exp(1.)

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_ackley_spd(spd_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Ackley function on the SPD manifold.
    Note: the based point is obtained from and can be modified in the function "get_ackley_base".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Ackley function on the SPD manifold
    :return opt_y: value of the global minimum of the Ackley function on the SPD manifold
    """
    # Optimum x
    opt_x = get_ackley_base(spd_manifold)
    opt_x_vec = symmetric_matrix_to_vector_mandel(opt_x)[None]
    # Optimum y
    opt_y = ackley_function_spd(torch.tensor(opt_x_vec), spd_manifold).numpy()

    return opt_x, opt_y


def get_ackley_base(spd_manifold):
    """
    This function returns the base point for the Ackley function on the SPD manifold.
    It guarantees that the base point is the same in the functions "ackley_function_spd" and
    "optimum_ackley_spd".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Ackley function on the SPD manifold
    :return opt_y: value of the global minimum of the Ackley function on the SPD manifold
    """
    return 2 * np.eye(spd_manifold._n)


def rosenbrock_function_spd(x, spd_manifold):
    """
    This function computes the Rosenbrock function on the SPD manifold.
    The Rosenbrock function is defined on the tangent space of a base point and projected to the manifold via the
    exponential map. The value of the function is therefore computed by projecting the point on the manifold to
    the tangent space of the base point and by computing the value of the Rosenbrock function in this Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array. The input is
    given in Mandel notation.

    Note: the based point is obtained from and can be modified in the function "get_rosenbrock_base".

    Parameters
    ----------
    :param x: point on the SPD manifold in Mandel notation      (torch tensor)
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Returns
    -------
    :return: value of the Rosenbrock function at x              (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Mandel notation dimension
    vector_dimension = int(dimension + dimension * (dimension - 1) / 2)

    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()
    if np.ndim(x) < 2:
        x = x[None]

    # To vector (Mandel notation)
    x = vector_to_symmetric_matrix_mandel(x[0])

    # Projection in tangent space of the mean.
    base = get_rosenbrock_base(spd_manifold)
    x_proj = spd_manifold.log(base, x)

    # Vectorize to use only once the symmetric elements
    # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
    x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
    x_proj_vec[dimension:] /= 2. ** 0.5

    # Rosenbrock function
    y = 0
    for i in range(vector_dimension - 1):
        y += 100 * (x_proj_vec[i + 1] - x_proj_vec[i] ** 2) ** 2 + (1 - x_proj_vec[i]) ** 2

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_rosenbrock_spd(spd_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Rosenbrock function on the SPD manifold.
    Note: the based point is obtained from and can be modified in the function "get_rosenbrock_base".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Rosenbrock function on the SPD manifold
    :return opt_y: value of the global minimum of the Rosenbrock function on the SPD manifold
    """
    # Dimension
    dimension = spd_manifold._n

    # Optimum x
    base = get_rosenbrock_base(spd_manifold)
    opt_x_log = np.ones((dimension, dimension))
    opt_x = spd_manifold.exp(base, opt_x_log)
    opt_x_vec = symmetric_matrix_to_vector_mandel(opt_x)[None]
    # Optimum y
    opt_y = rosenbrock_function_spd(torch.tensor(opt_x_vec), spd_manifold).numpy()

    return opt_x, opt_y


def get_rosenbrock_base(spd_manifold):
    """
    This function returns the base point for the Rosenbrock function on the SPD manifold.
    It guarantees that the base point is the same in the functions "rosenbrock_function_spd" and
    "optimum_rosenbrock_spd".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Rosenbrock function on the SPD manifold
    :return opt_y: value of the global minimum of the Rosenbrock function on the SPD manifold
    """
    return 2 * np.eye(spd_manifold._n)


def bimodal_function_spd(x, spd_manifold):
    """
    This function computes a bimodal Gaussian distribution on the SPD manifold.
    Each Gaussian is defined on the tangent space of their mean and projected to the manifold via the exponential map.
    The value of the function is therefore computed by projecting the point on the manifold to the tangent space of the
    means, computing the value of the function for each Gaussian in the corresponding Euclidean space and additioning
    the values obtained for each Gaussian
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array. The input is
    given in Mandel notation.

    Note: the means and covariances of the two Gaussians can be modified in the function "get_bimodal_parameters".

    Parameters
    ----------
    :param x: point on the SPD manifold in Mandel notation      (torch tensor)
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Returns
    -------
    :return: value of the bimodal distribution at x             (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Mandel notation dimension
    vector_dimension = int(dimension + dimension * (dimension - 1) / 2)

    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()

    if np.ndim(x) < 2:
        x = x[None]

    # To vector (Mandel notation)
    x = vector_to_symmetric_matrix_mandel(x[0])

    # Function parameters
    mu1, mu2, sigma1, sigma2 = get_bimodal_parameters(spd_manifold)
    # Useful operation values
    inv_sigma1 = np.linalg.inv(sigma1)
    det_sigma1 = np.linalg.det(sigma1)
    inv_sigma2 = np.linalg.inv(sigma2)
    det_sigma2 = np.linalg.det(sigma2)

    # Probability computation for each Gaussian
    # Gaussian 1
    x_proj1 = symmetric_matrix_to_vector_mandel(spd_manifold.log(mu1, x))
    prob1 = -np.exp(- 0.5 * np.dot(x_proj1, np.dot(inv_sigma1, x_proj1.T))) / \
            np.sqrt((2 * np.pi) ** vector_dimension * det_sigma1)
    # Gaussian 2
    x_proj2 = symmetric_matrix_to_vector_mandel(spd_manifold.log(mu2, x))
    prob2 = -np.exp(- 0.5 * np.dot(x_proj2, np.dot(inv_sigma2, x_proj2.T))) / \
            np.sqrt((2 * np.pi) ** vector_dimension * det_sigma2)

    # Function value
    y = prob1 + prob2

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_bimodal_spd(spd_manifold):
    """
    This function returns the global minimum (x, f(x)) of the bimodal distribution on the SPD manifold.
    Note: the means and covariances of the two Gaussians can be modified in the function "get_bimodal_parameters".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the bimodal distribution on the SPD manifold
    :return opt_y: value of the global minimum of the bimodal distribution on the SPD manifold
    """
    # Function parameters
    mu1, mu2, sigma1, sigma2 = get_bimodal_parameters(spd_manifold)

    # Test both means
    test_val1 = bimodal_function_spd(torch.tensor(symmetric_matrix_to_vector_mandel(mu1)), spd_manifold).numpy()
    test_val2 = bimodal_function_spd(torch.tensor(symmetric_matrix_to_vector_mandel(mu2)), spd_manifold).numpy()

    # Optimum x and y
    if test_val1 < test_val2:
        opt_x = mu1
        opt_y = test_val1
    else:
        opt_x = mu2
        opt_y = test_val2

    return opt_x, opt_y


def get_bimodal_parameters(spd_manifold):
    """
    This function returns the means and covariances of the two Gaussian for the bimodal distribution test function.
    It guarantees that the parameters are the same in the functions "bimodal_function_spd" and
    "optimum_bimodal_spd".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Returns
    -------
    :return mu1: mean of the first Gaussian distribution
    :return mu2: mean of the second Gaussian distribution
    :return sigma1: covariance of the first Gaussian distribution
    :return sigma2: covariance of the second Gaussian distribution
    """
    # Dimension
    dimension = spd_manifold._n
    # Mandel notation dimension
    vector_dimension = int(dimension + dimension * (dimension - 1) / 2)

    # Gaussian 1
    mu1 = 0.1 * np.ones((dimension, dimension))
    mu1 += 0.4 * np.eye(dimension)
    sigma1 = np.eye(vector_dimension) / 4.

    # Gaussian 2
    mu2 = -0.3 * np.ones((dimension, dimension))
    mu2 += 3.7 * np.eye(dimension)
    sigma2 = np.eye(vector_dimension) / 3.

    return mu1, mu2, sigma1, sigma2


def product_of_sines_function_spd(x, spd_manifold, coefficient=100.):
    """
    This function computes the Product of sines function on the SPD manifold.
    The Product of sines function is defined on the tangent space of a base point and projected to the manifold via the
    exponential map. The value of the function is therefore computed by projecting the point on the manifold to
    the tangent space of the base point and by computing the value of the function in this Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array. The input is
    given in Mandel notation.

    Note: the based point is obtained from and can be modified in the function "get_product_of_sines_base".

    Parameters
    ----------
    :param x: point on the SPD manifold (in Mandel notation)    (torch tensor)
    :param spd_manifold: n-dimensional SPD manifold             (pymanopt manifold)

    Optional parameters
    -------------------
    :param coefficient: multiplying coefficient of the product of sines

    Returns
    -------
    :return: value of the Product of sines function at x                  (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Mandel notation dimension
    vector_dimension = int(dimension + dimension * (dimension - 1) / 2)

    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()

    if np.ndim(x) < 2:
        x = x[None]

    # To vector (Mandel notation)
    x = vector_to_symmetric_matrix_mandel(x[0])

    # Projection in tangent space of the base
    base = get_product_of_sines_base(spd_manifold)
    x_proj = spd_manifold.log(base, x)

    # Vectorize to use only once the symmetric elements
    # Division by sqrt(2) to keep the original elements (this is equivalent to Voigt instead of Mandel)
    x_proj_vec = symmetric_matrix_to_vector_mandel(x_proj)
    x_proj_vec[dimension:] /= 2. ** 0.5

    # Sines
    sin_x_proj_vec = np.sin(x_proj_vec)

    # Product of sines function
    y = coefficient * sin_x_proj_vec[0] * np.prod(sin_x_proj_vec)

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_product_of_sines_spd(spd_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Product of sines function on the SPD manifold.
    Note: the based point is obtained from and can be modified in the function "get_product_of_sines_base".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Product of sines function on the SPD manifold
    :return opt_y: value of the global minimum of the Product of sines function on the SPD manifold
    """
    # Dimension
    dimension = spd_manifold._n

    # Optimum x
    base = get_product_of_sines_base(spd_manifold)
    opt_x_log = np.pi/2 * np.ones((dimension, dimension))
    opt_x_log[1, 1] = -np.pi/2
    opt_x = spd_manifold.exp(base, opt_x_log)
    opt_x_vec = symmetric_matrix_to_vector_mandel(opt_x)[None]
    # Optimum y
    opt_y = product_of_sines_function_spd(torch.tensor(opt_x_vec), spd_manifold).numpy()

    return opt_x, opt_y


def get_product_of_sines_base(spd_manifold):
    """
    This function returns the base point for the Product of sines function on the SPD manifold.
    It guarantees that the base point is the same in the functions "product_of_sines_function_spd" and
    "optimum_product_of_sines_spd".

    Parameters
    ----------
    :param spd_manifold: n-dimensional SPD manifold          (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Product of sines function on the SPD manifold
    :return opt_y: value of the global minimum of the Product of sines function on the SPD manifold
    """
    return 2 * np.eye(spd_manifold._n)


def cholesky_function_wrapped(x_cholesky, spd_manifold, test_function):
    """
    This function is a wrapper for tests function on the SPD manifold with inputs in the form of a Cholesky
    decomposition. The Cholesky decomposition input is transformed into the corresponding SPD matrix which is then
    given as input for the given test function.

    Parameters
    ----------
    :param x_cholesky: Cholesky decomposition of a SPD matrix
    :param spd_manifold: n-dimensional SPD manifold                     (pymanopt manifold)
    :param test_function: function on the SPD manifold to be tested

    Returns
    -------
    :return: value of the test function at x                            (numpy [1,1] array)
    """
    # Dimension
    dimension = spd_manifold._n

    # Data to numpy
    torch_type = x_cholesky.dtype
    x_cholesky = x_cholesky.detach().numpy()

    if np.ndim(x_cholesky) == 2:
        x_cholesky = x_cholesky[0]

    # Verify that Cholesky decomposition does not have zero
    if x_cholesky.size - np.count_nonzero(x_cholesky):
        x_cholesky += 1e-6

    # Add also a small value to too-close-to-zero Cholesky decomposition elements
    x_cholesky[np.abs(x_cholesky) < 1e-10] += 1e-10

    # Reshape matrix
    indices = np.tril_indices(dimension)
    xL = np.zeros((dimension, dimension))
    xL[indices] = x_cholesky

    # Compute SPD from Cholesky
    x = np.dot(xL, xL.T)
    # Mandel notation
    x = symmetric_matrix_to_vector_mandel(x)
    # To torch
    x = torch.from_numpy(x).to(dtype=torch_type)

    # Test function
    return test_function(x, spd_manifold)


