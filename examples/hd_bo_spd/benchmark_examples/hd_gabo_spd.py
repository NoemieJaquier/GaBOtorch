import numpy as np
import types
import random
import functools

import torch
import gpytorch
import botorch
from torch.autograd import Variable

import pymanopt.manifolds as pyman_man
import pymanopt.solvers as pyman_solvers

import matplotlib.pyplot as plt

from BoManifolds.Riemannian_utils.spd_utils import symmetric_matrix_to_vector_mandel, \
    vector_to_symmetric_matrix_mandel, spd_sample
from BoManifolds.Riemannian_utils.spd_utils_torch import symmetric_matrix_to_vector_mandel_torch, \
    vector_to_symmetric_matrix_mandel_torch

from BoManifolds.kernel_utils.kernels_spd import SpdLogEuclideanGaussianKernel
from BoManifolds.kernel_utils.kernels_nested_spd import NestedSpdLogEuclideanGaussianKernel

from BoManifolds.manifold_optimization.manifold_gp_fit import fit_gpytorch_manifold
from BoManifolds.manifold_optimization.manifold_optimize import joint_optimize_manifold
from BoManifolds.manifold_optimization.constrained_trust_regions import ConstrainedTrustRegions, \
    StrictConstrainedTrustRegions

from BoManifolds.nested_mappings.nested_spd_utils import projection_from_spd_to_nested_spd, \
    projection_from_nested_spd_to_spd
from BoManifolds.nested_mappings.nested_spd_optimization import optimize_reconstruction_parameters_nested_spd, \
    min_log_euclidean_distance_reconstruction_cost

from BoManifolds.nested_mappings.nested_spd_constraints_utils import max_eigenvalue_nested_spd_constraint, \
    min_eigenvalue_nested_spd_constraint, random_nested_spd_with_spd_eigenvalue_constraints

from BoManifolds.BO_test_functions.test_functions_spd import rosenbrock_function_spd, optimum_rosenbrock_spd
from BoManifolds.BO_test_functions.nested_test_functions_spd import projected_function_spd, \
    optimum_projected_function_spd

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

"""
This example shows the use of High-dimensional Geometry-aware Bayesian optimization (HD-GaBO) to optimize the Rosenbrock 
function defined on a low-dimensional SPD manifold S2_++ embedded in a high-dimensional SPD manifold S5_++. 

The test function, defined on the tangent space of the 2*I, is projected on the low-dimensional SPD manifold with the 
exponential map (i.e. the logarithm map is used to determine the function value). The function is then embedded in the 
high-dimensional SPD manifold.

The search space is defined as a subspace of the SPD manifold bounded by minimum and maximum eigenvalues. These bounds 
are defined by setting maximum/minimum eigenvalues to the pymanopt SPD manifold.

HD-GaBO uses a Gaussian kernel which uses a nested mapping to project the data from the high-dimensional SPD manifold
to the latent space (i.e., the low-dimensional SPD manifold) and computes the geodesic distance between the projected 
data.  To guarantee the positive-definiteness of the kernel, the lengthscale beta must be above the beta min value. 
This value can be determined by using the example kernels/spd_gaussian_kernel_parameters.py for each SPD manifold.
The acquisition function is optimized on the low-dimensional manifold with the constrained trust regions on Riemannian 
manifolds. The trust region algorithm is originally implemented in pymanopt. A constrained version is used here.
The next query point, obtained by the optimization on the low-dimensional manifold, is then projected back to the 
original space using a right-inverse of the nested mapping, whose parameters are optimized by minimizing the sum of 
squared residuals on the original manifold.

The dimension of the manifold and of the low-dimensional latent manifold are set by the variables 'dim' and 
'latent_dim', respectively. Note that the following element must be adapted when the dimension of the latent space is 
modified:
- beta_min must be recomputed for the new low-dimensional manifold;
The number of BO iterations is set by the user by changing the variable 'nb_iter_bo'.
The test function is the Rosenbrock function on the sphere, but can be changed by the user. Other test functions are 
available in BoManifolds.BO_test_functions.test_functions_spd.

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the spd manifold) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the 
    BO at each iteration. Note that the randomly generated initial data are not displayed, so that the iterations number 
    starts at the number of initial data + 1.

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""

if __name__ == "__main__":
    seed = 1234
    # Set numpy and pytorch seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Define the dimension
    dim = 5
    dim_vec = int(dim + dim*(dim-1)/2)
    latent_dim = 2

    # Beta min value
    if latent_dim == 2:
        beta_min = 0.6
    elif latent_dim == 3:
        beta_min = 0.2
    elif latent_dim == 5:
        beta_min = 0.25
    elif latent_dim == 7:
        beta_min = 0.22
    elif latent_dim == 10:
        beta_min = 0.2
    elif latent_dim == 12:
        beta_min = 0.16

    # Instantiate the manifolds
    spd_manifold = pyman_man.PositiveDefinite(dim)
    latent_spd_manifold = pyman_man.PositiveDefinite(latent_dim)

    # Update the random function of the manifold (the original one samples only eigenvalues between 1 and 2).
    # We need to specify the minimum and maximum eigenvalues of the random matrices. This is done when defining bounds.
    spd_manifold.rand = types.MethodType(spd_sample, spd_manifold)

    # Define the test function
    # Parameters for the nested test function
    grassmann_manifold = pyman_man.Grassmann(dim, latent_dim)
    projection_matrix_test = torch.from_numpy(grassmann_manifold.rand()).double()

    # Define the nested test function
    test_function = functools.partial(projected_function_spd, low_dimensional_spd_manifold=latent_spd_manifold,
                                      test_function=rosenbrock_function_spd,
                                      projection_matrix=projection_matrix_test)
    # Optimum
    true_min, true_opt_val = optimum_projected_function_spd(optimum_rosenbrock_spd, latent_spd_manifold,
                                                            projection_matrix_test)
    true_min_vec = symmetric_matrix_to_vector_mandel(true_min)[None]

    # Specify the optimization domain
    # Eigenvalue bounds
    min_eigenvalue = 1e-4
    max_eigenvalue = 5.
    # Manifolds eigenvalue bounds
    spd_manifold.min_eig = min_eigenvalue
    spd_manifold.max_eig = max_eigenvalue
    latent_spd_manifold.min_eig = min_eigenvalue
    latent_spd_manifold.max_eig = max_eigenvalue
    # Optimization domain
    lower_bound = torch.cat((min_eigenvalue * torch.ones(dim, dtype=torch.float64),
                             - max_eigenvalue / np.sqrt(2) * torch.ones(dim_vec - dim, dtype=torch.float64)))
    upper_bound = torch.cat((max_eigenvalue * torch.ones(dim, dtype=torch.float64),
                             max_eigenvalue / np.sqrt(2) * torch.ones(dim_vec - dim, dtype=torch.float64)))
    bounds = torch.stack([lower_bound, upper_bound])

    # Generate random data on the SPD manifold
    nb_data_init = 5
    x_init = np.array([spd_manifold.rand() for i in range(nb_data_init)])
    x_init_vec = np.array([symmetric_matrix_to_vector_mandel(x_init[i]) for i in range(nb_data_init)])
    x_data = torch.tensor(x_init)
    x_data_vec = torch.tensor(x_init_vec)
    y_data = torch.zeros(nb_data_init, dtype=torch.float64)
    for n in range(nb_data_init):
        y_data[n] = test_function(x_data_vec[n])

    # Define BO functions and parameters
    # Define the kernel functions
    k_fct = gpytorch.kernels.ScaleKernel(NestedSpdLogEuclideanGaussianKernel(dim=dim, latent_dim=latent_dim),
                                         outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    latent_k_fct = gpytorch.kernels.ScaleKernel(SpdLogEuclideanGaussianKernel(),
                                                outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0,
                                                                                                          0.15))
    latent_k_fct.to(x_data.dtype, non_blocking=False)  # Cast to type of x_data

    # Define the likelihood of the GPR models
    # A constant mean function is already included in the model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                          noise_constraint=
                                                                          gpytorch.constraints.GreaterThan(1e-8),
                                                                          initial_value=noise_prior_mode)
    latent_lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                                 noise_constraint=
                                                                                 gpytorch.constraints.GreaterThan(1e-8),
                                                                                 initial_value=noise_prior_mode)

    # Define the GPR model
    # A constant mean function is already included in the model
    model = botorch.models.SingleTaskGP(x_data_vec, y_data[:, None], covar_module=k_fct, likelihood=lik_fct)

    # Define the marginal log-likelihood
    mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Define the solver on the manifold
    projection_solver = pyman_solvers.ConjugateGradient(maxiter=50)
    reconstruction_solver = pyman_solvers.ConjugateGradient(maxiter=100)
    acquisition_solver = StrictConstrainedTrustRegions(mingradnorm=2e-4, maxiter=100, minstepsize=1e-4)

    # Initialize best observation and function value list
    new_best_f, index = y_data.min(0)
    best_x = [x_data_vec[index]]
    best_f = [new_best_f]

    # BO loop
    n_iters = 30
    for iteration in range(n_iters):
        # Fit GP model
        botorch.fit_gpytorch_model(mll=mll_fct, optimizer=fit_gpytorch_manifold, solver=projection_solver,
                                   nb_init_candidates=20)

        # Projection of the data into the low dimensional SPD manifold
        projection_matrix = Variable(k_fct.base_kernel.projection_matrix.data.clone(), requires_grad=False)
        # print(projection_matrix)

        x_data_projected_mat = projection_from_spd_to_nested_spd(x_data, projection_matrix).double()
        x_data_projected = symmetric_matrix_to_vector_mandel_torch(x_data_projected_mat)

        # Reproduce parameters for latent space model
        # latent_k_fct.base_kernel.beta = k_fct.base_kernel.beta
        latent_k_fct.base_kernel.lengthscale = k_fct.base_kernel.lengthscale
        latent_k_fct.outputscale = k_fct.outputscale
        latent_lik_fct.noise = lik_fct.noise
        if iteration == 0:
            # Create latent model
            latent_model = botorch.models.SingleTaskGP(x_data_projected, y_data[:, None],
                                                       covar_module=latent_k_fct, likelihood=latent_lik_fct)
        else:
            # Update latent model
            latent_model.set_train_data(x_data_projected, y_data,
                                        strict=False)  # strict False necessary to add datapoints

        # Optimize the parameters of the mapping from the latent to the ambient space
        projection_complement_matrix, bottom_spd_matrix, contraction_matrix = \
            optimize_reconstruction_parameters_nested_spd(x_data, x_data_projected_mat, projection_matrix,
                                                          reconstruction_solver,
                                                          cost_function=min_log_euclidean_distance_reconstruction_cost)

        # In order to obtain (with the acquisition function) a SPD matrix in the latent space that,
        # projected back in the ambient space, respects the constraints, we update the random and constraints
        # functions:
        # Update the random function of the manifold of nested SPD
        random_function_spd_manifold_red = functools.partial(random_nested_spd_with_spd_eigenvalue_constraints,
                                                             random_spd_fct=spd_manifold.rand,
                                                             projection_matrix=projection_matrix)
        latent_spd_manifold.rand = types.MethodType(random_function_spd_manifold_red, latent_spd_manifold)
        # Define the constraints for the optimization of the acquisition function
        max_eigenvalue_constraint = functools.partial(max_eigenvalue_nested_spd_constraint,
                                                      maximum_eigenvalue=max_eigenvalue,
                                                      projection_matrix=projection_matrix,
                                                      projection_complement_matrix=projection_complement_matrix,
                                                      bottom_spd_matrix=bottom_spd_matrix,
                                                      contraction_matrix=contraction_matrix)
        min_eigenvalue_constraint = functools.partial(min_eigenvalue_nested_spd_constraint,
                                                      minimum_eigenvalue=min_eigenvalue,
                                                      projection_complement_matrix=projection_complement_matrix,
                                                      projection_matrix=projection_matrix,
                                                      bottom_spd_matrix=bottom_spd_matrix,
                                                      contraction_matrix=contraction_matrix)
        inequality_constraints = [max_eigenvalue_constraint, min_eigenvalue_constraint]

        # Define the acquisition function
        acq_fct = botorch.acquisition.ExpectedImprovement(model=latent_model, best_f=best_f[-1], maximize=False)

        # Get new candidate
        new_x_projected = joint_optimize_manifold(acq_fct, latent_spd_manifold, acquisition_solver, q=1,
                                                  num_restarts=5,
                                                  raw_samples=100, bounds=bounds,
                                                  pre_processing_manifold=vector_to_symmetric_matrix_mandel_torch,
                                                  post_processing_manifold=symmetric_matrix_to_vector_mandel_torch,
                                                  approx_hessian=True,
                                                  inequality_constraints=inequality_constraints)

        # To matrix
        new_x_projected = vector_to_symmetric_matrix_mandel_torch(new_x_projected)

        # Projection back onto the original SPD manifold
        new_x = projection_from_nested_spd_to_spd(new_x_projected, projection_matrix, projection_complement_matrix,
                                                  bottom_spd_matrix, contraction_matrix)
        # To vector
        new_x_vec = symmetric_matrix_to_vector_mandel_torch(new_x)

        # Get new observation
        new_y = test_function(new_x_vec)[0]

        # Update training points
        x_data = torch.cat((x_data, new_x))
        x_data_vec = torch.cat((x_data_vec, new_x_vec))
        y_data = torch.cat((y_data, new_y))

        # Update best observation
        new_best_f, index = y_data.min(0)
        best_x.append(x_data_vec[index])
        best_f.append(new_best_f)

        # Update the model
        model.set_train_data(x_data_vec, y_data, strict=False)  # strict False necessary to add datapoints

        print("Iteration " + str(iteration) + "\t Best f " + str(new_best_f.item()))

    # To numpy
    x_eval = x_data.numpy()
    x_eval_vec = x_data_vec.numpy()
    y_eval = y_data.numpy()[:, None]

    # Compute distances between consecutive x's and best evaluation for each iteration
    neval = x_eval.shape[0]
    distances = np.zeros(neval-1)
    for n in range(neval-1):
        distances[n] = spd_manifold.dist(x_eval[n + 1, :], x_eval[n, :])

    y_best = np.ones(neval)
    for i in range(neval):
        y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(neval - 1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    plt.grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(neval)), y_best, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    plt.grid(True)

    plt.show()
