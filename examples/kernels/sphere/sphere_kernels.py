import numpy as np
import torch
import gpytorch
import botorch

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.Riemannian_utils.sphere_utils import logmap
from BoManifolds.kernel_utils.kernels_sphere import SphereGaussianKernel, SphereLaplaceKernel

from BoManifolds.plot_utils.manifolds_plots import plot_sphere

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
"""
This example shows the use of different kernels for the hypershere manifold S^n , used for Gaussian process regression.
The tested function corresponds to a Gaussian distribution with a mean defined on the sphere and a covariance defined on 
the tangent space of the mean. Training data are generated "far" from the mean. The trained Gaussian process is then 
used to determine the value of the function from test data sampled around the mean of the test function. 
The kernels used are:
    - Manifold-RBF kernel (geometry-aware)
    - Laplace kernel (geometry-aware)
    - Euclidean kernel (classical geometry-unaware)
    
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com
"""


def test_function(x, mu_test_function):
    x_proj = logmap(x, mu_test_function)

    sigma_test_fct = np.array([[0.6, 0.2, 0], [0.2, 0.3, -0.01], [0, -0.01, 0.2]])
    inv_sigma_test_fct = np.linalg.inv(sigma_test_fct)
    det_sigma_test_fct = np.linalg.det(sigma_test_fct)

    return np.exp(- 0.5 * np.dot(x_proj.T, np.dot(inv_sigma_test_fct, x_proj))) / np.sqrt(
        (2 * np.pi) ** dim * det_sigma_test_fct)


def plot_gaussian_process_prediction(figure_handle, mu, test_data, mean_est, mu_test_fct, title):
    ax = Axes3D(figure_handle)

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
    # Plot training data on the manifold
    plt_scale_fact = test_function(mu_test_fct, mu_test_fct)[0, 0]
    nb_data_test = test_data.shape[0]
    for n in range(nb_data_test):
        ax.scatter(test_data[n, 0], test_data[n, 1], test_data[n, 2],
                   c=[pl.cm.inferno(mean_est[n] / plt_scale_fact)])

    # Plot mean of Gaussian test function
    ax.scatter(mu[0], mu[1], mu[2], c='g', marker='D')
    plt.title(title, size=25)


if __name__ == "__main__":
    np.random.seed(1234)

    # Define the test function
    mu_test_fct = np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0])

    # Generate random data on the sphere
    nb_data = 20
    dim = 3

    mean = np.array([1, 0, 0])
    mean = mean / np.linalg.norm(mean)
    fact_cov = 0.1
    cov = fact_cov * np.eye(dim)

    data = np.random.multivariate_normal(mean, cov, nb_data)
    x_man = data / np.linalg.norm(data, axis=1)[:, None]

    y_train = np.zeros((nb_data, 1))
    for n in range(nb_data):
        y_train[n] = test_function(x_man[n], mu_test_fct)

    # Generate test data on the sphere
    nb_data_test = 10

    # mean_test = np.array([-2, 1, 0])
    # mean_test = np.array([2, 1, 0])
    mean_test = mu_test_fct
    mean_test = mean_test / np.linalg.norm(mean)
    fact_cov = 0.1
    cov_test = fact_cov * np.eye(dim)

    data = np.random.multivariate_normal(mean_test, cov_test, nb_data_test)
    x_man_test = data / np.linalg.norm(data, axis=1)[:, None]

    y_test = np.zeros((nb_data_test, 1))
    for n in range(nb_data_test):
        y_test[n] = test_function(x_man_test[n], mu_test_fct)

    # Plot training data - 3D figure
    fig = plt.figure(figsize=(5, 5))
    y_train_for_plot = y_train.reshape((len(y_train),))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man, y_train_for_plot, mu_test_fct, r'Training data')

    # Plot true test data
    # 3D figure
    fig = plt.figure(figsize=(5, 5))
    y_test_for_plot = y_test.reshape((len(y_test),))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, y_test_for_plot, mu_test_fct, r'Test data (ground truth)')

    # ### Gaussian kernel
    # Define the kernel
    k_gauss = gpytorch.kernels.ScaleKernel(SphereGaussianKernel(beta_min=6.5),
                                           outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_gauss = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                            noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                            initial_value=noise_prior_mode)
    m_gauss = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y_train),
                                          covar_module=k_gauss, likelihood=lik_gauss)
    # Define the marginal log-likelihood
    mll_gauss = gpytorch.mlls.ExactMarginalLogLikelihood(m_gauss.likelihood, m_gauss)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_gauss)
    # Kernel computation
    K1 = k_gauss.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_gauss.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_gauss.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_gauss = m_gauss(torch.tensor(x_man_test))
    mean_est_gauss = preds_gauss.mean.detach().numpy()
    var_est_gauss = preds_gauss.variance.detach().numpy()
    covar_est_gauss = preds_gauss.covariance_matrix.detach().numpy()
    # Compute posterior samples
    # posterior_samples = preds_gauss.sample(sample_shape=torch.Size(1000,))
    error_gauss = np.sqrt(np.sum((y_test - mean_est_gauss) ** 2) / nb_data_test)
    print('Estimation error (Manifold-RBF kernel) = ', error_gauss)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_gauss, mu_test_fct, r'Manifold-RBF kernel')

    # ### Laplace kernel
    # Define the kernel
    k_laplace = gpytorch.kernels.ScaleKernel(SphereLaplaceKernel(),
                                             outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))

    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_laplace = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                              noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                              initial_value=noise_prior_mode)
    m_laplace = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y_train),
                                            covar_module=k_laplace, likelihood=lik_laplace)
    # Define the marginal log-likelihood
    mll_laplace = gpytorch.mlls.ExactMarginalLogLikelihood(m_laplace.likelihood, m_laplace)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_laplace)
    # Kernel computation
    K1 = k_laplace.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_laplace.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_laplace.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_laplace = m_laplace(torch.tensor(x_man_test))
    mean_est_laplace = preds_laplace.mean.detach().numpy()
    var_est_laplace = preds_laplace.variance.detach().numpy()
    covar_est_laplace = preds_laplace.covariance_matrix.detach().numpy()
    error_laplace = np.sqrt(np.sum((y_test - mean_est_laplace) ** 2) / nb_data_test)
    print('Estimation error (Laplace kernel) = ', error_laplace)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_laplace, mu_test_fct, r'Laplace kernel')

    # ### Euclidean RBF
    # Define the kernel
    k_eucl = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=None),
                                          outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_eucl = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                           initial_value=noise_prior_mode)
    m_eucl = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y_train),
                                         covar_module=k_eucl, likelihood=lik_eucl)
    # Define the marginal log-likelihood
    mll_eucl = gpytorch.mlls.ExactMarginalLogLikelihood(m_eucl.likelihood, m_eucl)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_eucl)
    # Kernel computation
    K1 = k_eucl.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_eucl.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_eucl.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_eucl = m_eucl(torch.tensor(x_man_test))
    mean_est_eucl = preds_eucl.mean.detach().numpy()
    var_est_eucl = preds_eucl.variance.detach().numpy()
    covar_est_eucl = preds_eucl.covariance_matrix.detach().numpy()
    error_eucl = np.sqrt(np.sum((y_test - mean_est_eucl) ** 2) / nb_data_test)
    print('Estimation error (Euclidean-RBF kernel) = ', error_eucl)
    # Plot test data
    fig = plt.figure(figsize=(5, 5))
    plot_gaussian_process_prediction(fig, mu_test_fct, x_man_test, mean_est_eucl, mu_test_fct, r'Euclidean-RBF kernel')

    plt.show()

