import numpy as np
import types
import random
import functools

import torch
import gpytorch
import botorch

import pymanopt.manifolds as pyman_man

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.euclidean_optimization.euclidean_constrained_optimize import joint_optimize

from BoManifolds.Riemannian_utils.spd_utils import symmetric_matrix_to_vector_mandel, spd_sample
from BoManifolds.Riemannian_utils.spd_utils_torch import vector_to_symmetric_matrix_mandel_torch
from BoManifolds.Riemannian_utils.spd_constraints_utils import min_eigenvalue_constraint_cholesky, \
    max_eigenvalue_constraint_cholesky
from BoManifolds.Riemannian_utils.spd_constraints_utils_torch import post_processing_spd_cholesky_torch

from BoManifolds.plot_utils.manifolds_plots import plot_spd_cone
from BoManifolds.plot_utils.bo_plots import bo_plot_function_spd, bo_plot_gp_spd, bo_plot_acquisition_spd

from BoManifolds.BO_test_functions.test_functions_spd import ackley_function_spd, optimum_ackley_spd, \
    cholesky_function_wrapped

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

'''
This example shows the use of Cholesky Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function.
An Euclidean BO is applied on the Cholesky decomposition of the SPD matrices. 

The test function, defined on the tangent space of the north pole, is projected on the SPD manifold with the 
exponential map (i.e. the logarithm map is used to determine the function value). 
The search space is defined as a subspace of the SPD manifold bounded by minimum and maximum eigenvalues. 
The Euclidean BO uses a Gaussian kernel for comparisons with GaBO.
The acquisition function is optimized with on the Cholesky decomposition to obtain points lying on the SPD manifold. 
The domain of the optimization is defined for the Cholesky decomposition. Constraints on minimum and maximum eigenvalues 
are added to satisfy the bound constraints on the eigenvalues.

The dimension of the manifold is set by the variable 'dim'. Note that the following element must be adapted when the 
dimension is modified:
- if the dimension is not 3, 'display_figures' is set to 'False'.
The number of BO iterations is set by the user by changing the variable 'nb_iter_bo'.
The test function is the Ackley function on the sphere, but can be changed by the user. Other test functions are 
available in BoManifolds.BO_test_functions.test_functions_spd.

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the spd manifold) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the 
    BO at each iteration. Note that the randomly generated initial data are not displayed, so that the iterations number 
    starts at the number of initial data + 1.
The following graphs are produced by this example if 'display_figures' is 'True':
- the true function graph is displayed on S2_++;
- the acquisition function at the end of the optimization is displayed on S2_++;
- the GP mean at the end of the optimization is displayed on S2_++;
- the GP mean and variances are displayed on 2D projections of S2_++;
- the BO observations are displayed on S2_++.
For all the graphs, the optimum parameter is displayed with a star, the current best estimation with a diamond and all 
the BO observation with dots.

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
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
    dim = 2
    dim_vec = int(dim + dim*(dim-1)/2)

    if dim == 2:
        disp_fig = True
    else:
        disp_fig = False

    # Instantiate the manifold
    spd_manifold = pyman_man.PositiveDefinite(dim)

    # Update the random function of the manifold (the original one samples only eigenvalues between 1 and 2).
    # We need to specify the minimum and maximum eigenvalues of the random matrices. This is done when defining bounds.
    spd_manifold.rand = types.MethodType(spd_sample, spd_manifold)

    # Function to optimize
    test_function_chol = functools.partial(cholesky_function_wrapped, test_function=ackley_function_spd,
                                           spd_manifold=spd_manifold)
    # Optimum
    true_min, true_opt_val = optimum_ackley_spd(spd_manifold)
    true_min_vec = symmetric_matrix_to_vector_mandel(true_min)[None]
    true_min_chol = np.linalg.cholesky(true_min)
    true_min_chol = true_min_chol[np.tril_indices(dim)]

    # Plot test function with inputs on the sphere
    # 3D figure
    r_cone = 5.
    if disp_fig:
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)

        max_colors = bo_plot_function_spd(ax, test_function_chol, r_cone=r_cone, true_opt_x=true_min,
                                          true_opt_y=true_opt_val, alpha=0.3, n_elems=100, n_elems_h=10, chol=True)
        ax.set_title('True function', fontsize=20)
        plt.show()
    else:
        max_colors = None

    # Specify the optimization domain
    min_eigenvalue = 0.001
    max_eigenvalue = 5.
    spd_manifold.min_eig = min_eigenvalue
    spd_manifold.max_eig = max_eigenvalue
    lower_bound = - np.sqrt(max_eigenvalue) * torch.ones(dim_vec, dtype=torch.float64)
    upper_bound = np.sqrt(max_eigenvalue) * torch.ones(dim_vec, dtype=torch.float64)
    # j = 0
    # a = 2
    # while j < dim_vec:
    #     lower_bound[j] = 0.
    #     j += a
    #     a += 1
    bounds = torch.stack([lower_bound, upper_bound])

    # Define constraints
    min_eigenvalue_ineq_function = functools.partial(min_eigenvalue_constraint_cholesky,
                                                     min_eigenvalue=min_eigenvalue)
    max_eigenvalue_ineq_function = functools.partial(max_eigenvalue_constraint_cholesky,
                                                     max_eigenvalue=max_eigenvalue)
    constraints = [{'type': 'ineq', 'fun': lambda x: min_eigenvalue_ineq_function(x)},
                   {'type': 'ineq', 'fun': lambda x: max_eigenvalue_ineq_function(x)}]

    # Post processing function
    post_processing_fct = functools.partial(post_processing_spd_cholesky_torch, min_eigenvalue=min_eigenvalue,
                                            max_eigenvalue=max_eigenvalue)

    # Generate random data on the SPD manifold
    nb_data_init = 5

    x_init_mat = np.array([spd_manifold.rand() for i in range(nb_data_init)])
    x_init_chol = np.array([np.linalg.cholesky(x_init_mat[i])[np.tril_indices(dim)] for i in range(nb_data_init)])
    x_data = torch.tensor(x_init_chol)

    y_data = torch.zeros(nb_data_init, dtype=torch.float64)
    for n in range(nb_data_init):
        y_data[n] = test_function_chol(x_data[n])

    # Define the kernel function
    # With Cholesky, a lengthscale prior is often needed for a good execution.
    k_fct = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.torch_priors.GammaPrior(3.0, 6.0)),
        outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))

    # Define the GPR model
    # A constant mean function is already included in the model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                          noise_constraint=
                                                                          gpytorch.constraints.GreaterThan(1e-8),
                                                                          initial_value=noise_prior_mode)
    model = botorch.models.SingleTaskGP(x_data, y_data[:, None], covar_module=k_fct, likelihood=lik_fct)

    # Define the marginal log-likelihood
    mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Initialize best observation and function value list
    new_best_f, index = y_data.min(0)
    best_x_chol = [x_data[index]]
    best_f = [new_best_f]

    # BO loop
    n_iters = 50
    for iteration in range(n_iters):
        # Fit GP model
        botorch.fit_gpytorch_model(mll=mll_fct)

        # Define the acquisition function
        acq_fct = botorch.acquisition.ExpectedImprovement(model=model, best_f=best_f[-1], maximize=False)

        # Get new candidate
        new_x = joint_optimize(acq_fct, bounds=bounds, q=1, num_restarts=5, raw_samples=100, constraints=constraints,
                               post_processing_init=post_processing_fct)

        # Get new observation
        new_y = test_function_chol(new_x)[0]

        # Update training points
        x_data = torch.cat((x_data, new_x))
        y_data = torch.cat((y_data, new_y))

        # Update best observation
        new_best_f, index = y_data.min(0)
        best_x_chol.append(x_data[index])
        best_f.append(new_best_f)

        # Update the model
        model.set_train_data(x_data, y_data, strict=False)  # strict False necessary to add datapoints

        print("Iteration " + str(iteration) + "\t Best f " + str(new_best_f.item()))

    # To numpy
    x_eval_chol = x_data.numpy()
    y_eval = y_data.numpy()[:, None]

    # To vector/matrix representation
    x_eval_mat = np.zeros((x_eval_chol.shape[0], dim, dim))
    x_eval = np.zeros((x_eval_chol.shape[0], dim_vec))
    indices = np.tril_indices(dim)
    for n in range(x_eval_chol.shape[0]):
        sigmaL = np.zeros((dim, dim))
        sigmaL[indices] = x_eval_chol[n]
        x_eval_mat[n] = np.dot(sigmaL, sigmaL.T)
        x_eval[n] = symmetric_matrix_to_vector_mandel(x_eval_mat[n])

    best_x_mat = np.zeros((len(best_x_chol), dim, dim))
    best_x = np.zeros((len(best_x_chol), dim_vec))
    for n in range(len(best_x_chol)):
        sigmaL = np.zeros((dim, dim))
        sigmaL[indices] = best_x_chol[n].detach().numpy()
        best_x_mat[n] = np.dot(sigmaL, sigmaL.T)
        best_x[n] = symmetric_matrix_to_vector_mandel(best_x_mat[n])

    if disp_fig:
        # Plot acquisition function
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        bo_plot_acquisition_spd(ax, acq_fct, r_cone=r_cone, xs=x_eval, opt_x=best_x[-1][None],
                                true_opt_x=true_min_vec, n_elems=20, n_elems_h=10)
        ax.set_title('Acquisition function', fontsize=20)
        plt.show()

        # Plot GP
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        bo_plot_gp_spd(ax, model, r_cone=r_cone, xs=x_eval, opt_x=best_x[-1][None], true_opt_x=true_min_vec,
                       true_opt_y=true_opt_val, max_colors=25., n_elems=20,
                       n_elems_h=10)
        ax.set_title('GP mean', fontsize=20)
        plt.show()

    if dim == 2:
        # Plot convergence on the sphere
        # 3D figure
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)

        # Make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        # Remove axis
        ax._axis3don = False

        # Initial view
        ax.view_init(elev=10, azim=-20.)  # (default: elev=30, azim=-60)
        # ax.view_init(elev=10, azim=50.)  # (default: elev=30, azim=-60)

        # Plot SPD cone
        plot_spd_cone(ax, r=r_cone, lim_fact=0.8)

        # Plot evaluated points
        if max_colors is None:
            max_colors = np.max(y_eval - true_opt_val[0])
        for n in range(x_eval.shape[0]):
            ax.scatter(x_eval[n, 0], x_eval[n, 1], x_eval[n, 2]/np.sqrt(2.),
                       c=pl.cm.inferno(1. - (y_eval[n] - true_opt_val[0]) / max_colors))

        # Plot true minimum
        ax.scatter(true_min[0, 0], true_min[1, 1], true_min[0, 1], s=40, c='g', marker='P')

        # Plot BO minimum
        ax.scatter(best_x[-1][0], best_x[-1][1], best_x[-1][2] / np.sqrt(2.), s=20, c='r', marker='D')
        ax.set_title('BO observations', fontsize=20)
        plt.show()

    # Compute distances between consecutive x's and best evaluation for each iteration
    neval = x_eval.shape[0]
    distances = np.zeros(neval-1)
    for n in range(neval-1):
        distances[n] = np.linalg.norm(x_eval_chol[n + 1, :] - x_eval_chol[n, :])

    y_best = np.ones(neval)
    for i in range(neval):
        y_best[i] = y_eval[:(i + 1)].min()

    #  Plot distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.array(range(neval - 1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive observations')
    plt.grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(np.array(range(neval)), y_best, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    plt.grid(True)

    plt.show()
