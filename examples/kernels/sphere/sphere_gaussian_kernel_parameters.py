import numpy as np
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.kernel_utils.kernels_sphere import SphereGaussianKernel

from BoManifolds.plot_utils.manifolds_plots import plot_sphere

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
"""
This example shows the experimental selection of parameters for the Sphere Gaussian kernel. To do so, a random sampling 
is carried out from different Gaussian distributions on the manifold (random mean and identity covariance 
matrix). 
After, the corresponding kernel matrix is computed for a range of values for $beta$, with $theta = 1$. This process is 
repeated several times (in this case, 10) for each value of $beta$. 
A minimum value of $beta$ is set to the lowest $beta$ value leading to all the kernel matrices to be positive-definite.

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com 
"""


# Align y axis for double y axis plot
def align_y_axis(axis1, v1, axis2, v2):
    _, y1 = axis1.transData.transform((0, v1))
    _, y2 = axis2.transData.transform((0, v2))
    inv = axis2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = axis2.get_ylim()
    axis2.set_ylim(miny+dy, maxy+dy)


if __name__ == "__main__":
    # Generate random data in the manifold
    dim = 3  # Dimension of the manifold
    nb_samples = 500  # Total number of samples
    nb_sources = 10  # Number of Gaussians to sample from
    nb_trials = 20  # Number of set of random points to test
    fact_cov = 1.  # Do not put it too big, otherwise the projected data on the manifold can be really far

    # Origin in the manifold
    origin_man = np.array([0, 0, 1])

    # Define the range of parameter for the kernel
    nb_params = 30
    if dim == 3:
        betas = np.logspace(0, 5, nb_params)
    elif dim == 4:
        betas = np.logspace(0, 2, nb_params)
    elif 5 <= dim <= 10:
        betas = np.logspace(-0.2, 1.5, nb_params)
    elif dim > 10 :
        betas = np.logspace(-1.5, 0.5, nb_params)

    min_eigval_trials = []

    for trial in range(nb_trials):
        print('Trial ', trial)

        # Means and covariances to generate random data
        mean = [np.random.randn(dim) for i in range(nb_sources)]
        mean = np.array(mean)
        mean = mean / np.linalg.norm(mean, axis=1)[:, None]
        cov = fact_cov * np.eye(dim)

        # This way of sampling is not extremely rigorous.
        # The sampling should be done on the tangent space and projected on the manifold.
        # However, it should be sufficient for the current purpose.
        # Sample data
        data = [np.random.multivariate_normal(mean[i], cov, int(nb_samples / nb_sources)).T for i in range(nb_sources)]

        # Project samples on the manifold
        data_man_tmp = []
        for i in range(nb_sources):
            for n in range(int(nb_samples / nb_sources)):
                data_man_tmp.append(data[i][:, n] / np.linalg.norm(data[i][:, n])[None])
        data_man = np.array(data_man_tmp)
        nb_data = data_man.shape[0]

        # Define and compute the kernel for the parameters
        K = []
        min_eigval = []

        for i in range(nb_params):
            # Create kernel instance and set beta
            k = SphereGaussianKernel(beta_min=0.0)
            k.beta = betas[i]

            # Compute the kernel
            Ktmp = k.forward(torch.tensor(data_man), torch.tensor(data_man)).detach().numpy()
            K.append(Ktmp)

            # Compute the eigenvalues
            eigvals, _ = np.linalg.eig(Ktmp)
            eigvals = np.real(eigvals)
            min_eigval.append(np.min(eigvals))

        # Minimum eigenvalue of the kernel
        min_eigval = np.array(min_eigval)

        min_eigval_trials.append(min_eigval)

    # Compute percentage of PD kernels
    pd_kernels = np.array(min_eigval_trials)
    pd_kernels[pd_kernels > 0] = 1.
    pd_kernels[pd_kernels <= 0] = 0.
    percentage_pd_kernels = np.sum(pd_kernels, axis=0) / nb_trials

    print(betas)
    print(percentage_pd_kernels)

    # Plot input data if dim is 3
    if dim == 3:
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

        # Plot sphere
        plot_sphere(ax, alpha=0.4)

        # Plot training data on the manifold
        plt.plot(data_man[:, 0], data_man[:, 1], data_man[:, 2], color='k', marker='.', linewidth=0., markersize=3.)

        # Plot mean of generated data
        plt.plot(mean[:, 0], mean[:, 1], mean[:, 2], color='r', marker='.', linewidth=0., markersize=6.)

        plt.title(r'Training data', size=20)

    # Plot minimum eigenvalue in function of the kernel parameter
    min_eigval_trials = np.array(min_eigval_trials)
    min_eigval_mean = np.mean(min_eigval_trials, axis=0)
    min_eigval_std = np.std(min_eigval_trials, axis=0)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    plt.fill_between(np.log10(betas), min_eigval_mean - min_eigval_std, min_eigval_mean + min_eigval_std, alpha=0.2,
                     color='orchid')
    plt.plot(np.log10(betas), min_eigval_mean, marker='o', color='orchid')
    plt.plot(np.log10(betas), np.zeros(nb_params), color='k')
    ax.set_xlabel(r'$\log_{10}(\beta)$')
    ax.set_ylabel(r'$\lambda_{\min}(\bm{K})$')

    # Plot percentage of positive kernel in function of the kernel parameter
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    plt.plot(np.log10(betas), percentage_pd_kernels, marker='o', color='darkblue')
    plt.plot(np.log10(betas), np.zeros(nb_params), color='k')
    ax.set_xlabel(r'$\log_{10}(\beta)$')
    ax.set_ylabel(r'PD percentage of $\bm{K}$')
    plt.show()

    # Plot min eigenvalue and percentage of PD kernel in function of the kernel parameter (one graph)
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(np.log10(betas), min_eigval_mean, color='orchid', marker='o')
    ax1.fill_between(np.log10(betas), min_eigval_mean - min_eigval_std, min_eigval_mean + min_eigval_std, color='orchid', alpha=0.2)
    ax2.plot(np.log10(betas), percentage_pd_kernels*100, color='darkblue', marker='o')
    ax2.plot(np.log10(betas), np.zeros(nb_params), color='k')

    ax1.tick_params(labelsize=30)
    ax2.tick_params(labelsize=30)
    ax1.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='y', nbins=4)

    ax1.set_xlabel(r'$\log_{10}(\beta)$', fontsize=44)
    ax1.set_ylabel(r'$\lambda_{\min}(\bm{K})$', fontsize=44)
    ax2.set_ylabel(r'PD percentage of $\bm{K}$', fontsize=44)

    align_y_axis(ax1, 0, ax2, 0)

    filename = '../../../Figures/sphere' + str(dim-1) + '_kernel_params.png'
    plt.savefig(filename, bbox_inches='tight')
