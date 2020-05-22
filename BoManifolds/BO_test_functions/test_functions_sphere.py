import numpy as np
import torch

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2019
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
'''


def ackley_function_sphere(x, sphere_manifold):
    """
    This function computes the Ackley function on the sphere manifold.
    The Ackley function is defined on the tangent space of the base point (1, 0, 0, ...) and projected to the manifold
    via the exponential map. The value of the function is therefore computed by projecting the point on the manifold to
    the tangent space of the base point and by computing the value of the Ackley function in this Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array.

    Note: the base point was chosen as (1, 0, 0, ...) for simplicity. As the first coordinate of Log_base(x) is
    always 0, the vectors in the tangent space can easily be expressed in a n-1 dimensional space by ignoring the first
    coordinate and the Ackley function is computed in this space. If the base point is modified, the computation of the
    function must be updated.

    Parameters
    ----------
    :param x: point on the sphere                               (torch tensor)
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return: value of the Ackley function at x                  (numpy [1,1] array)
    """
    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()
    if np.ndim(x) < 2:
        x = x[None]

    # Dimension of the manifold
    dimension = sphere_manifold._shape[0]

    # Projection in tangent space of the mean.
    # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
    # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
    # dimensional space by simply ignoring the first coordinate.
    base = np.zeros((1, dimension))
    base[0, 0] = 1.
    x_proj = sphere_manifold.log(base, x)[0]

    # Remove first dim
    x_proj_red = x_proj[1:]
    reduced_dimension = dimension-1

    # Ackley function parameters
    a = 20
    b = 0.2
    c = 2*np.pi

    # Ackley function
    aexp_term = -a * np.exp(-b * np.sqrt(np.sum(x_proj_red**2) / reduced_dimension))
    expcos_term = - np.exp( np.sum(np.cos(c*x_proj_red) / reduced_dimension))
    y = aexp_term + expcos_term + a + np.exp(1.)

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_ackley_sphere(sphere_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Ackley function on the sphere manifold.
    Note: the base point must be fixed as in the function "ackley_function_sphere" as (1, 0, 0, ...).

    Parameters
    ----------
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Ackley function on the sphere
    :return opt_y: value of the global minimum of the Ackley function on the sphere
    """
    # Dimension
    dimension = sphere_manifold._shape[0]

    # Optimum x
    opt_x = np.zeros((1, dimension))
    opt_x[0, 0] = 1
    # Optimum y
    opt_y = ackley_function_sphere(torch.tensor(opt_x), sphere_manifold).numpy()

    return opt_x, opt_y


def rosenbrock_function_sphere(x, sphere_manifold):
    """
    This function computes the Rosenbrock function on the sphere manifold.
    The Rosenbrock function is defined on the tangent space of the base point (1, 0, 0, ...) and projected to the
    manifold via the exponential map. The value of the function is therefore computed by projecting the point on the
    manifold to the tangent space of the base point and by computing the value of the Rosenbrock function in this
    Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array.

    Note: the base point was chosen as (1, 0, 0, ...) for simplicity. As the first coordinate of Log_base(x) is
    always 0, the vectors in the tangent space can easily be expressed in a n-1 dimensional space by ignoring the first
    coordinate and the function is computed in this space. If the base point is modified, the computation of the
    function must be updated.

    Parameters
    ----------
    :param x: point on the sphere                               (torch tensor)
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return: value of the Rosenbrock function at x              (numpy [1,1] array)
    """
    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()
    if np.ndim(x) < 2:
        x = x[None]

    # Dimension of the manifold
    dimension = sphere_manifold._shape[0]

    # Projection in tangent space of the mean.
    # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
    # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
    # dimensional space by simply ignoring the first coordinate.
    base = np.zeros((1, dimension))
    base[0, 0] = 1.
    x_proj = sphere_manifold.log(base, x)[0]

    # Remove first dim
    x_proj_red = x_proj[1:]
    # reduced_dimension = dimension - 1

    # Rosenbrock function
    # y = 0
    # for i in range(reduced_dimension - 1):
    #     y += 100 * (x_proj_red[i + 1] - x_proj_red[i] ** 2) ** 2 + (1 - x_proj_red[i]) ** 2

    a = (x_proj_red[1:] - x_proj_red[:-1] ** 2)
    b = (1 - x_proj_red[:-1])
    y = np.sum(100 * a * a + b * b)

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_rosenbrock_sphere(sphere_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Rosenbrock function on the sphere manifold.
    Note: the base point must be fixed as in the function "rosenbrock_function_sphere" as (1, 0, 0, ...).

    Parameters
    ----------
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Rosenbrock function on the sphere
    :return opt_y: value of the global minimum of the Rosenbrock function on the sphere
    """
    # Dimension
    dimension = sphere_manifold._shape[0]

    # Optimum x
    base = np.zeros((1, dimension))
    base[0, 0] = 1
    opt_x_log = np.ones((1, dimension))
    opt_x = sphere_manifold.exp(base, opt_x_log)
    # Optimum y
    opt_y = rosenbrock_function_sphere(torch.tensor(opt_x), sphere_manifold).numpy()

    return opt_x, opt_y


def bimodal_function_sphere(x, sphere_manifold):
    """
    This function computes a bimodal Gaussian distribution on the sphere manifold.
    Each Gaussian is defined on the tangent space of their mean and projected to the manifold via the exponential map.
    The value of the function is therefore computed by projecting the point on the manifold to the tangent space of the
    means, computing the value of the function for each Gaussian in the corresponding Euclidean space and additioning
    the values obtained for each Gaussian
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array.

    Note: the means and covariances of the two Gaussians can be modified in the function "get_bimodal_parameters".

    Parameters
    ----------
    :param x: point on the sphere                               (torch tensor)
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return: value of the bimodal distribution at x             (numpy [1,1] array)
    """
    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()

    if np.ndim(x) < 2:
        x = x[None]

    # Dimension of the manifold
    dimension = sphere_manifold._shape[0]

    # Function parameters
    mu1, mu2, sigma1, sigma2 = get_bimodal_parameters(sphere_manifold)
    # Useful operation values
    inv_sigma1 = np.linalg.inv(sigma1)
    det_sigma1 = np.linalg.det(sigma1)
    inv_sigma2 = np.linalg.inv(sigma2)
    det_sigma2 = np.linalg.det(sigma2)

    # Probability computation for each Gaussian
    # Gaussian 1
    x_proj1 = sphere_manifold.log(mu1, x)
    prob1 = -np.exp(- 0.5 * np.dot(x_proj1, np.dot(inv_sigma1, x_proj1.T))) / \
            np.sqrt((2 * np.pi) ** dimension * det_sigma1)
    # Gaussian 2
    x_proj2 = sphere_manifold.log(mu2, x)
    prob2 = -np.exp(- 0.5 * np.dot(x_proj2, np.dot(inv_sigma2, x_proj2.T))) / \
            np.sqrt((2 * np.pi) ** dimension * det_sigma2)

    # Function value
    y = prob1 + prob2

    return torch.tensor(y, dtype=torch_type)


def optimum_bimodal_sphere(sphere_manifold):
    """
    This function returns the global minimum (x, f(x)) of the bimodal distribution on the sphere manifold.
    Note: the means and covariances of the two Gaussians can be modified in the function "get_bimodal_parameters".

    Parameters
    ----------
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the bimodal distribution on the sphere
    :return opt_y: value of the global minimum of the bimodal distribution on the sphere
    """
    # Function parameters
    mu1, mu2, sigma1, sigma2 = get_bimodal_parameters(sphere_manifold)

    # Test both means
    test_val1 = bimodal_function_sphere(torch.tensor(mu1), sphere_manifold).numpy()
    test_val2 = bimodal_function_sphere(torch.tensor(mu2), sphere_manifold).numpy()

    # Optimum x and y
    if test_val1 < test_val2:
        opt_x = mu1
        opt_y = test_val1
    else:
        opt_x = mu2
        opt_y = test_val2

    return opt_x, opt_y


def get_bimodal_parameters(sphere_manifold):
    """
    This function returns the means and covariances of the two Gaussian for the bimodal distribution test function.
    It guarantees that the parameters are the same in the functions "bimodal_function_sphere" and
    "optimum_bimodal_sphere".

    Parameters
    ----------
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return mu1: mean of the first Gaussian distribution
    :return mu2: mean of the second Gaussian distribution
    :return sigma1: covariance of the first Gaussian distribution
    :return sigma2: covariance of the second Gaussian distribution
    """
    # Dimension
    dimension = sphere_manifold._shape[0]

    # Gaussian 1
    mu1 = np.zeros((1, dimension))
    if dimension == 2:
        mu1[0, 0] = 1.
    else:
        mu1[0, 0] = 0.5
        mu1[0, 1] = 0.5
        mu1[0, 2] = -1. / np.sqrt(2)
    sigma1 = np.eye(dimension) / 2.5

    # Gaussian 2
    mu2 = np.ones((1, dimension)) / np.sqrt(dimension)
    sigma2 = np.eye(dimension) / 3.

    return mu1, mu2, sigma1, sigma2


def product_of_sines_function_sphere(x, sphere_manifold, coefficient=100):
    """
    This function computes the Product of sines function on the sphere manifold.
    The function is defined on the tangent space of the base point (1, 0, 0, ...) and projected to the
    manifold via the exponential map. The value of the function is therefore computed by projecting the point on the
    manifold to the tangent space of the base point and by computing the value of the Rosenbrock function in this
    Euclidean space.
    To be used for BO, the input x is a torch tensor and the function must output a numpy [1,1] array.

    Note: the base point was chosen as (1, 0, 0, ...) for simplicity. As the first coordinate of Log_base(x) is
    always 0, the vectors in the tangent space can easily be expressed in a n-1 dimensional space by ignoring the first
    coordinate and the function is computed in this space. If the base point is modified, the computation of the
    function must be updated.

    Parameters
    ----------
    :param x: point on the sphere                               (torch tensor)
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Optional parameters
    -------------------
    :param coefficient: multiplying coefficient of the product of sines

    Returns
    -------
    :return: value of the Product of sines function at x        (numpy [1,1] array)
    """
    # Data to numpy
    torch_type = x.dtype
    x = x.detach().numpy()
    if np.ndim(x) < 2:
        x = x[None]

    # Dimension of the manifold
    dimension = sphere_manifold._shape[0]

    # Projection in tangent space of the mean.
    # The base is fixed at (1, 0, 0, ...) for simplicity. Therefore, the tangent plane is aligned with the axis x.
    # The first coordinate of x_proj is always 0, so that vectors in the tangent space can be expressed in a dim-1
    # dimensional space by simply ignoring the first coordinate.
    base = np.zeros((1, dimension))
    base[0, 0] = 1.
    x_proj = sphere_manifold.log(base, x)[0]

    # Remove first dim
    x_proj_red = x_proj[1:]
    sin_x_proj_red = np.sin(x_proj_red)

    # Product of sines function
    y = coefficient * sin_x_proj_red[0] * np.prod(sin_x_proj_red)

    return torch.tensor(y[None, None], dtype=torch_type)


def optimum_product_of_sines_sphere(sphere_manifold):
    """
    This function returns the global minimum (x, f(x)) of the Product of sines function on the sphere manifold.
    Note: the base point must be fixed as in the function "product_of_sines_function_sphere" as (1, 0, 0, ...).

    Parameters
    ----------
    :param sphere_manifold: n-dimensional sphere manifold       (pymanopt manifold)

    Returns
    -------
    :return opt_x: location of the global minimum of the Styblinsky-Tank function on the sphere
    :return opt_y: value of the global minimum of the Styblinsky-Tank function on the sphere
    """
    # Dimension
    dimension = sphere_manifold._shape[0]

    # Optimum x
    base = np.zeros((1, dimension))
    base[0, 0] = 1
    # opt_x = np.zeros((1, dimension))
    # opt_x[0, 0] = 1
    opt_x_log = np.pi/2. * np.ones((1, dimension))
    opt_x_log[0, 0] = 1
    opt_x_log[0, 2] = - np.pi/2.
    opt_x = sphere_manifold.exp(base, opt_x_log)
    # Optimum y
    opt_y = product_of_sines_function_sphere(torch.tensor(opt_x), sphere_manifold).numpy()

    return opt_x, opt_y

