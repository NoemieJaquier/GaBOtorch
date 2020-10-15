import numpy as np
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
from BoManifolds.Riemannian_utils.sphere_constraint_utils import norm_one_constraint

from BoManifolds.plot_utils.manifolds_plots import plot_sphere
from BoManifolds.plot_utils.bo_plots import bo_plot_function_sphere, bo_plot_acquisition_sphere, bo_plot_gp_sphere, \
    bo_plot_gp_sphere_planar

from BoManifolds.BO_test_functions.test_functions_sphere import ackley_function_sphere, optimum_ackley_sphere

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'

'''
This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function. 

The test function, defined on the tangent space of the north pole, is projected on the sphere with the exponential 
map (i.e. the logarithm map is used to determine the function value). 
The Euclidean BO uses a Gaussian kernel for comparisons with GaBO.
The acquisition function is optimized with a constrained optimization to obtain points lying on the sphere.
The dimension of the manifold is set by the variable 'dim'. Note that the following element must be adapted when the 
dimension is modified:
- if the dimension is not 3, 'display_figures' is set to 'False'.
The number of BO iterations is set by the user by changing the variable 'nb_iter_bo'.
The test function is the Ackley function on the sphere, but can be changed by the user. Other test functions are 
available in BoManifolds.BO_test_functions.test_functions_sphere.

The current optimum value of the function is printed at each BO iteration and the optimal estimate of the optimizer 
(on the sphere) is printed at the end of the queries. 
The following graphs are produced by this example:
- the convergence graph shows the distance between two consecutive iterations and the best function value found by the 
    BO at each iteration. Note that the randomly generated initial data are not displayed, so that the iterations number 
    starts at the number of initial data + 1.
The following graphs are produced by this example if 'display_figures' is 'True':
- the true function graph is displayed on S2;
- the acquisition function at the end of the optimization is displayed on S2;
- the GP mean at the end of the optimization is displayed on S2;
- the GP mean and variances are displayed on 2D projections of S2;
- the BO observations are displayed on S2.
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
    dim = 3

    if dim == 3:
        disp_fig = True
    else:
        disp_fig = False

    # Instantiate the manifold (used for the test function)
    sphere_manifold = pyman_man.Sphere(dim)

    # Function to optimize
    test_function = functools.partial(ackley_function_sphere, sphere_manifold=sphere_manifold)
    # Optimum
    true_min, true_opt_val = optimum_ackley_sphere(sphere_manifold)

    # Plot test function with inputs on the sphere
    # 3D figure
    if disp_fig:
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)

        max_colors = bo_plot_function_sphere(ax, test_function, true_opt_x=true_min, true_opt_y=true_opt_val,
                                             elev=10, azim=30, n_elems=300)
        ax.set_title('True function', fontsize=20)
        plt.show()
    else:
        max_colors = None

    # Generate random data on the sphere
    nb_data_init = 5
    x_data = torch.tensor(np.array([sphere_manifold.rand() for n in range(nb_data_init)]))
    y_data = torch.zeros(nb_data_init, dtype=torch.float64)
    for n in range(nb_data_init):
        y_data[n] = test_function(x_data[n])

    # Define the kernel function
    k_fct = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(),
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

    # Specify the optimization domain
    bounds = torch.stack([-torch.ones(dim, dtype=torch.float64), torch.ones(dim, dtype=torch.float64)])

    # Initialize best observation and function value list
    new_best_f, index = y_data.min(0)
    best_x = [x_data[index]]
    best_f = [new_best_f]

    # Define constraints
    constraints = [{'type': 'eq', 'fun': norm_one_constraint}]

    # Define sampling post processing function
    def post_processing_init(x):
        return x / torch.cat(x.shape[-1] * [torch.norm(x, dim=[-1]).unsqueeze(-1)], dim=-1)

    # BO loop
    n_iters = 25
    for iteration in range(n_iters):
        # Fit GP model
        botorch.fit_gpytorch_model(mll=mll_fct)

        # Define the acquisition function
        acq_fct = botorch.acquisition.ExpectedImprovement(model=model, best_f=best_f[-1], maximize=False)

        # Get new candidate
        new_x = joint_optimize(acq_fct, bounds=bounds, q=1, num_restarts=5, raw_samples=100, constraints=constraints,
                               post_processing_init=post_processing_init)

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

    if disp_fig:
        # Plot acquisition function
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        bo_plot_acquisition_sphere(ax, acq_fct, xs=x_eval, opt_x=best_x[-1][None], true_opt_x=true_min,
                                   elev=10, azim=30, n_elems=100)
        ax.set_title('Acquisition function', fontsize=20)
        plt.show()

        # Plot GP
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        bo_plot_gp_sphere(ax, model, xs=x_eval, opt_x=best_x[-1][None], true_opt_x=true_min, true_opt_y=true_opt_val,
                          max_colors=max_colors, elev=10, azim=30, n_elems=100)
        ax.set_title('GP mean', fontsize=20)
        plt.show()

        # Plot GP projected on planes
        fig = plt.figure(figsize=(10, 5))
        bo_plot_gp_sphere_planar(fig, model, var_fact=2., xs=x_eval, ys=y_eval, opt_x=best_x[-1][None],
                                 opt_y=best_f[-1], true_opt_x=true_min, true_opt_y=true_opt_val, max_colors=max_colors,
                                 n_elems=100)
        plt.title('GP mean and variance', fontsize=20)
        plt.show()

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
    # ax.view_init(elev=10, azim=-20.)  # (default: elev=30, azim=-60)
    ax.view_init(elev=10, azim=30.)  # (default: elev=30, azim=-60)

    # Plot sphere
    plot_sphere(ax, alpha=0.4)

    # Plot evaluated points
    if max_colors is None:
        max_colors = np.max(y_eval - true_opt_val[0])
    for n in range(x_eval.shape[0]):
        ax.scatter(x_eval[n, 0], x_eval[n, 1], x_eval[n, 2],
                   c=pl.cm.inferno(1. - (y_eval[n] - true_opt_val[0]) / max_colors))

    # Plot true minimum
    ax.scatter(true_min[0, 0], true_min[0, 1], true_min[0, 2], s=40, c='g', marker='P')

    # Plot BO minimum
    ax.scatter(best_x[-1][0], best_x[-1][1], best_x[-1][2], s=20, c='r', marker='D')
    ax.set_title('BO observations', fontsize=20)
    plt.show()

    # Compute distances between consecutive x's and best evaluation for each iteration
    neval = x_eval.shape[0]
    distances = np.zeros(neval-1)
    for n in range(neval-1):
        distances[n] = np.linalg.norm(x_eval[n + 1, :] - x_eval[n, :])

    Y_best = np.ones(neval)
    for i in range(neval):
        Y_best[i] = y_eval[:(i + 1)].min()

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
    plt.plot(np.array(range(neval)), Y_best, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    plt.grid(True)

    plt.show()
