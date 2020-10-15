import numpy as np
import random
import functools

import torch
import gpytorch
import botorch
from torch.autograd import Variable

import pymanopt.manifolds as pyman_man
import pymanopt.solvers as pyman_solvers

import matplotlib.pyplot as plt

from BoManifolds.kernel_utils.kernels_sphere import SphereGaussianKernel
from BoManifolds.kernel_utils.kernels_nested_sphere import NestedSphereGaussianKernel

from BoManifolds.manifold_optimization.manifold_gp_fit import fit_gpytorch_manifold
from BoManifolds.manifold_optimization.manifold_optimize import joint_optimize_manifold

from BoManifolds.nested_mappings.nested_spheres_utils import projection_from_sphere_to_subsphere, \
    projection_from_subsphere_to_sphere

from BoManifolds.nested_mappings.nested_spheres_optimization import \
    optimize_reconstruction_parameters_nested_sphere

from BoManifolds.BO_test_functions.test_functions_sphere import ackley_function_sphere, optimum_ackley_sphere
from BoManifolds.BO_test_functions.nested_test_functions_sphere import nested_function_sphere, \
    optimum_nested_function_sphere

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

'''
This example shows the use of High-dimensional Geometry-aware Bayesian optimization (HD-GaBO) to optimize the Ackley 
function defined on a low-dimensional sphere S2 embedded in a high-dimensional sphere S5. 

The test function, defined on the tangent space of the north pole, is projected on the low-dimensional sphere with the 
exponential map (i.e. the logarithm map is used to determine the function value). The function is then embedded in the 
high-dimensional sphere.
HD-GaBO uses a Gaussian kernel which uses a nested mapping to project the data from the high-dimensional sphere to the 
latent space (i.e., the low-dimensional sphere) and computes the geodesic distance between the projected data. 
The parameters of the nested mapping, the kernel parameters, and other Gaussian process parameters are optimized 
jointly. To guarantee the positive-definiteness of the kernel, the lengthscale beta must be above the beta min value of 
the low-dimensional sphere. This value can be determined by using the example 
kernels/sphere_gaussian_kernel_parameters.py for each latent sphere manifold.
The acquisition function is optimized on the low-dimensional manifold with trust regions on Riemannian manifolds, 
originally implemented in pymanopt. A robust version is used here to avoid crashing if NaNs or zero values occur during 
the optimization.
The next query point, obtained by the optimization on the low-dimensional manifold, is then projected back to the 
original space using a right-inverse of the nested mapping, whose parameters are optimized by minimizing the sum of 
squared residuals on the original manifold.

The dimension of the manifold and of the low-dimensional latent manifold are set by the variables 'dim' and 
'latent_dim', respectively. Note that the following element must be adapted when the dimension of the latent space is 
modified:
- beta_min must be recomputed for the new low-dimensional manifold;
The number of BO iterations is set by the user by changing the variable 'nb_iter_bo'.
The test function is the Ackley function on the sphere, but can be changed by the user. Other test functions are 
available in BoManifolds.BO_test_functions.test_functions_sphere.

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the sphere) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the
    BO at each iteration. Note that the randomly generated initial data are not displayed, so that the iterations number 
    starts at the number of initial data + 1.

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
'''

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

    # Define the latent space dimension
    latent_dim = 3

    # Instantiate the manifold
    sphere_manifold = pyman_man.Sphere(dim)
    latent_sphere_manifold = pyman_man.Sphere(latent_dim)

    # Define the test function
    # Parameters for the nested test function
    sphere_axes_test = [torch.zeros(dim - d, dtype=torch.float64) for d in range(0, dim - latent_dim)]
    for d in range(0, dim - latent_dim):
        sphere_axes_test[d][0] = 1.
    sphere_distances_to_axes_test = [torch.tensor(np.pi / 4, dtype=torch.float64) for d in range(0, dim - latent_dim)]
    # Nested test function
    test_function = functools.partial(nested_function_sphere, subsphere_manifold=latent_sphere_manifold,
                                      test_function=ackley_function_sphere, sphere_axes=sphere_axes_test,
                                      sphere_distances_to_axes=sphere_distances_to_axes_test)
    # Optimum
    true_min, true_opt_val = optimum_nested_function_sphere(optimum_ackley_sphere, latent_sphere_manifold,
                                                            sphere_axes_test, sphere_distances_to_axes_test)

    # Generate random data on the sphere
    nb_data_init = 5
    x_data = torch.tensor(np.array([sphere_manifold.rand() for n in range(nb_data_init)]), dtype=torch.float64)
    y_data = torch.zeros(nb_data_init, dtype=torch.float64)
    for n in range(nb_data_init):
        y_data[n] = test_function(x_data[n])

    # Define beta_min
    if latent_dim == 3:
        beta_min = 6.5
    elif latent_dim == 4:
        beta_min = 2.
    elif latent_dim == 5:
        beta_min = 1.2
    elif latent_dim == 6:
        beta_min = 1.0
    elif latent_dim == 11:
        beta_min = 0.6
    elif latent_dim == 21:
        beta_min = 0.35
    elif latent_dim == 51:
        beta_min = 0.21
    elif latent_dim == 101:
        beta_min = 0.21

    # Define the kernel functions
    k_fct = gpytorch.kernels.ScaleKernel(NestedSphereGaussianKernel(dim=dim, latent_dim=latent_dim, beta_min=beta_min),
                                         outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    latent_k_fct = gpytorch.kernels.ScaleKernel(SphereGaussianKernel(beta_min=beta_min),
                                                outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    latent_k_fct.to(x_data.dtype, non_blocking=False)  # Cast to type of x_data

    # Define the likelihood functions
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
    model = botorch.models.SingleTaskGP(x_data, y_data[:, None], covar_module=k_fct, likelihood=lik_fct)

    # Define the marginal log-likelihood
    mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Specify the optimization domain
    bounds = torch.stack([-torch.ones(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)])

    # Define the solver on the manifold
    reconstruction_solver = pyman_solvers.TrustRegions()
    solver = pyman_solvers.TrustRegions()

    # Initialize best observation and function value list
    new_best_f, index = y_data.min(0)
    best_x = [x_data[index]]
    best_f = [new_best_f]

    # BO loop
    nb_iter_bo = 25
    for iteration in range(nb_iter_bo):
        # Fit GP model
        # botorch.fit_gpytorch_model(mll=mll_fct)
        # botorch.fit_gpytorch_model(mll=mll_fct, optimizer=botorch.optim.fit.fit_gpytorch_torch)
        botorch.fit_gpytorch_model(mll=mll_fct, optimizer=fit_gpytorch_manifold)

        # Projection of the data into the low dimensional SPD manifold
        sphere_axes = [Variable(axis.data.clone(), requires_grad=False) for axis in k_fct.base_kernel.axes]
        sphere_distances_to_axis = [Variable(distance.data.clone(), requires_grad=False) for distance in
                                    k_fct.base_kernel.distances_to_axis]
        x_subsphere = projection_from_sphere_to_subsphere(x_data, sphere_axes, sphere_distances_to_axis)[-1]

        # Reproduce for latent space model
        latent_k_fct.base_kernel.beta = k_fct.base_kernel.beta
        latent_k_fct.outputscale = k_fct.outputscale
        latent_lik_fct.noise = lik_fct.noise
        if iteration == 0:
            # Create latent model
            latent_model = botorch.models.SingleTaskGP(x_subsphere, y_data[:, None],
                                                       covar_module=latent_k_fct, likelihood=latent_lik_fct)
        else:
            # Update latent model
            latent_model.set_train_data(x_subsphere, y_data, strict=False)  # strict False necessary to add datapoints

        # Optimize the parameters of the mapping from the latent to the ambient space
        sphere_distances_to_axis = optimize_reconstruction_parameters_nested_sphere(x_data, x_subsphere,
                                                                                    sphere_axes,
                                                                                    reconstruction_solver)

        # Define the acquisition function
        acq_fct = botorch.acquisition.ExpectedImprovement(model=latent_model, best_f=best_f[-1], maximize=False)

        # Get new candidate
        new_x_subsphere = joint_optimize_manifold(acq_fct, latent_sphere_manifold, solver, q=1,
                                                  num_restarts=5, raw_samples=100,
                                                  bounds=bounds)

        # Projection back onto the original sphere
        new_x = projection_from_subsphere_to_sphere(new_x_subsphere, sphere_axes, sphere_distances_to_axis)[-1]

        # Get new observation
        new_y = test_function(new_x)[0]

        # Update training points
        x_data = torch.cat((x_data, new_x))
        y_data = torch.cat((y_data, new_y))

        # Update best observation
        new_best_f, index = y_data.min(0)
        best_x.append(x_data[index])
        best_f.append(new_best_f)

        # Update the model
        model.set_train_data(x_data, y_data, strict=False)  # strict False necessary to add datapoints

        print("Iteration " + str(iteration) + "\t Best f " + str(new_best_f.item()))

    # To numpy
    x_eval = x_data.numpy()
    y_eval = y_data.numpy()[:, None]

    # Compute distances between consecutive x's and best evaluation for each iteration
    neval = x_eval.shape[0]
    distances = np.zeros(neval-1)
    for n in range(neval-1):
        distances[n] = sphere_manifold.dist(x_eval[n + 1, :], x_eval[n, :])

    Y_best = np.ones(neval)
    for i in range(neval):
        Y_best[i] = y_eval[:(i + 1)].min()

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
    plt.plot(np.array(range(neval)), Y_best, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    plt.grid(True)

    plt.show()
