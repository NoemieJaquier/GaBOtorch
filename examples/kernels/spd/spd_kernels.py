import numpy as np
from scipy.io import loadmat
import torch
import gpytorch
import botorch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.kernel_utils.kernels_spd import SpdAffineInvariantGaussianKernel, SpdFrobeniusGaussianKernel, \
    SpdLogEuclideanGaussianKernel
from BoManifolds.Riemannian_utils.spd_utils import expmap, symmetric_matrix_to_vector_mandel, \
    vector_to_symmetric_matrix_mandel

from BoManifolds.plot_utils.manifolds_plots import plot_spd_cone

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
"""
This example shows the use of different kernels for the SPD manifold, used for Gaussian process regression.
Artificial data are created from the time t and positions x of C-shape trajectory. The input data corresponds to the 
symmetric matrices xx' projected to the SPD manifold and the output data correspond to the time. Only a part of the 
trajectory is used to form the training data, while the whole trajectory is used to form the test data. 
Gaussian processes are trained on the training data and used to predict the output of the test data.
The kernels used are:
    - Affine-Invariant Gaussian Kernel (geometry-aware)
    - Frobenius Gaussian Kernel 
    - Log-Euclidean kernel 

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com
"""


def plot_training_test_data_spd_cone(training_spd_data, test_spd_data, figure_handle):
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
    ax.view_init(elev=10, azim=-20.)  # (default: elev=30, azim=-60)
    # ax.view_init(elev=10, azim=50.)  # (default: elev=30, azim=-60)

    # Plot SPD cone
    plot_spd_cone(ax, r=2.5, lim_fact=0.8)

    # Plot testing data on the manifold
    plt.plot(test_spd_data[0], test_spd_data[1], test_spd_data[2] / np.sqrt(2), color='b', marker='.', linewidth=0.,
             markersize=9.)

    # Plot training data on the manifold
    plt.plot(training_spd_data[0], training_spd_data[1], training_spd_data[2] / np.sqrt(2), color='k', marker='.',
             linewidth=0., markersize=9.)

    plt.title('Input data', size=25)


def plot_gaussian_process(true_output, mean, variance, posterior_samples, plot_title):
    number_data = true_output.shape[0]
    t = np.array(range(0, number_data))
    plt.figure(figsize=(12, 6))
    plt.plot(t, true_output, 'kx', mew=2)
    plt.plot(t, mean, 'C0', lw=2)
    plt.fill_between(t, mean - 1.96 * np.sqrt(variance), mean + 1.96 * np.sqrt(variance), color='C0', alpha=0.2)

    for s in range(nb_samples_post):
        plt.plot(t, posterior_samples[s], 'C0', linewidth=0.5)

    plt.title(plot_title, size=25)


if __name__ == "__main__":
    # Load data from 2D letters
    nb_samples = 1

    data_demos = loadmat('../../../data/2Dletters/C.mat')
    data_demos = data_demos['demos'][0]
    demos = [data_demos[i]['pos'][0][0] for i in range(data_demos.shape[0])]

    # Number of samples, time sampling
    nb_data_init = demos[0].shape[1]
    dt = 1.

    time = np.hstack([np.arange(0, nb_data_init) * dt] * data_demos.shape[0])
    demos_np = np.hstack(demos)

    # Euclidean vector data
    data_eucl = np.vstack((time, demos_np))
    data_eucl = data_eucl[:, :nb_data_init * nb_samples]

    # Create artificial SPD matrices from demonstrations and store them in Mandel notation (along with time)
    data_spd_mandel = [symmetric_matrix_to_vector_mandel(expmap(0.01 * np.dot(data_eucl[1:, n][:, None],
                                                                              data_eucl[1:, n][None]),
                                                                np.eye(2)))[:, None] for n in range(data_eucl.shape[1])]
    data_spd_mandel = np.vstack((data_eucl[0], np.concatenate(data_spd_mandel, axis=1)))

    # Training data
    data = data_spd_mandel[:, ::2]
    # Removing data to show GP uncertainty
    # id_to_remove = np.hstack((np.arange(12, 27), np.arange(34, 38)))
    # id_to_remove = np.hstack((np.arange(24, 54), np.arange(68, 76)))
    id_to_remove = np.hstack((np.arange(24, 37), np.arange(68, 76)))
    # id_to_remove = np.hstack((np.arange(12, 24), np.arange(76, 84)))
    data = np.delete(data, id_to_remove, axis=1)
    nb_data = data.shape[1]
    dim = 2
    dim_vec = 3

    # Training data in SPD form
    y = data[0][:, None]
    x_man = data[1:]
    x_man_mat = np.zeros((nb_data, dim, dim))
    for n in range(nb_data):
        x_man_mat[n] = vector_to_symmetric_matrix_mandel(x_man[:, n])

    # New output vector
    y_test = data_spd_mandel[0, ::2][:, None]
    nb_data_test = y_test.shape[0]

    # Test data in SPD form
    x_man_test = data_spd_mandel[1:, ::2]
    x_man_mat_test = np.zeros((nb_data_test, dim, dim))
    for n in range(nb_data_test):
        x_man_mat_test[n] = vector_to_symmetric_matrix_mandel(x_man_test[:, n])

    # Plot input data
    # 3D figure
    fig = plt.figure(figsize=(5, 5))
    plot_training_test_data_spd_cone(x_man, x_man_test, fig)

    # Transpose of input data
    x_man = x_man.T
    x_man_test = x_man_test.T

    # ### Affine invariant kernel
    # Define the kernel
    k_ai = gpytorch.kernels.ScaleKernel(SpdAffineInvariantGaussianKernel(beta_min=0.6),
                                        outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_ai = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                         noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                         initial_value=noise_prior_mode)
    m_ai = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y), covar_module=k_ai, likelihood=lik_ai)
    # Define the marginal log-likelihood
    mll_ai = gpytorch.mlls.ExactMarginalLogLikelihood(m_ai.likelihood, m_ai)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_ai)
    # Evaluation of kernel for input-input, input-test, and test-test
    K1 = k_ai.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_ai.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_ai.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_ai = m_ai(torch.tensor(x_man_test))
    mean_ai = preds_ai.mean.detach().numpy()
    var_ai = preds_ai.variance.detach().numpy()
    cov_ai = preds_ai.covariance_matrix.detach().numpy()
    # Compute posterior samples
    nb_samples_post = 10
    posterior_samples_ai = preds_ai.sample(torch.Size([nb_samples_post])).numpy()
    # Plot
    plot_gaussian_process(y_test, mean_ai, var_ai, posterior_samples_ai, 'Affine-invariant kernel')

    # ### Frobenius kernel
    # Define the kernel
    k_frob = gpytorch.kernels.ScaleKernel(SpdFrobeniusGaussianKernel(),
                                          outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_frob = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                           initial_value=noise_prior_mode)
    m_frob = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y), covar_module=k_frob, likelihood=lik_frob)
    # Define the marginal log-likelihood
    mll_frob = gpytorch.mlls.ExactMarginalLogLikelihood(m_frob.likelihood, m_frob)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_frob)
    # Evaluation of kernel for input-input, input-test, and test-test
    K1 = k_frob.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_frob.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_frob.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_frob = m_frob(torch.tensor(x_man_test))
    mean_frob = preds_frob.mean.detach().numpy()
    var_frob = preds_frob.variance.detach().numpy()
    cov_frob = preds_frob.covariance_matrix.detach().numpy()
    # Compute posterior samples
    posterior_samples_frob = preds_frob.sample(torch.Size([nb_samples_post])).numpy()
    # Plot
    plot_gaussian_process(y_test, mean_frob, var_frob, posterior_samples_frob, 'Frobenius kernel')

    # ### Log-Euclidean kernel
    # Define the kernel
    k_loge = gpytorch.kernels.ScaleKernel(SpdLogEuclideanGaussianKernel(),
                                          outputscale_prior=gpytorch.priors.torch_priors.GammaPrior(2.0, 0.15))
    # GPR model
    noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    lik_loge = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=noise_prior,
                                                                           noise_constraint=gpytorch.constraints.GreaterThan(1e-8),
                                                                           initial_value=noise_prior_mode)
    m_loge = botorch.models.SingleTaskGP(torch.tensor(x_man), torch.tensor(y), covar_module=k_loge, likelihood=lik_loge)
    # Define the marginal log-likelihood
    mll_loge = gpytorch.mlls.ExactMarginalLogLikelihood(m_loge.likelihood, m_loge)
    # Optimization of the model parameters
    botorch.fit_gpytorch_model(mll=mll_loge)
    # Evaluation of kernel for input-input, input-test, and test-test
    K1 = k_loge.forward(torch.tensor(x_man), torch.tensor(x_man))
    K12 = k_loge.forward(torch.tensor(x_man), torch.tensor(x_man_test))
    K2 = k_loge.forward(torch.tensor(x_man_test), torch.tensor(x_man_test))
    # Prediction
    preds_loge = m_loge(torch.tensor(x_man_test))
    mean_loge = preds_loge.mean.detach().numpy()
    var_loge = preds_loge.variance.detach().numpy()
    cov_loge = preds_loge.covariance_matrix.detach().numpy()
    # Compute posterior samples
    posterior_samples_loge = preds_loge.sample(torch.Size([nb_samples_post])).numpy()
    # Plot
    plot_gaussian_process(y_test, mean_loge, var_loge, posterior_samples_loge, 'Log-Euclidean kernel')

    plt.show()
