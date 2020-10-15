import types
import numpy as np
import torch

import pymanopt.manifolds as pyman_man

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from BoManifolds.Riemannian_utils.spd_utils import symmetric_matrix_to_vector_mandel, \
    vector_to_symmetric_matrix_mandel, spd_sample
from BoManifolds.kernel_utils.kernels_spd import SpdAffineInvariantGaussianKernel

from BoManifolds.plot_utils.manifolds_plots import plot_spd_cone

plt.rcParams['text.usetex'] = True  # use Latex font for plots
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
"""
This example shows the experimental selection of parameters for the SPD Affine-Invariant kernel. To do so, a random 
sampling is carried out on the SPD manifold for minimum and maximum eigenvalues included in a given interval.
After, the corresponding kernel matrix is computed for a range of values for $beta$, with $theta = 1$. This process is 
repeated several times (in this case, 10) for each value of $beta$. 
A minimum value of $beta$ is set to the lowest $beta$ value leading to all the kernel matrices to be positive-definite. 

This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com 
"""


# Align yaxis for double y axis plot
def align_y_axis(axis1, v1, axis2, v2):
    _, y1 = axis1.transData.transform((0, v1))
    _, y2 = axis2.transData.transform((0, v2))
    inv = axis2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = axis2.get_ylim()
    axis2.set_ylim(miny + dy, maxy + dy)


if __name__ == "__main__":
    # Generate random data in the manifold
    dim = 3  # Dimension of the manifold
    nb_samples = 500  # Total number of samples
    nb_sources = 10  # Number of Gaussians to sample from
    nb_trials = 10  # Number of set of random points to test
    fact_cov = 1.  # Do not put it too big, otherwise the projected data on the manifold can be really far

    # Minimum tolerated eigenvalue
    # We add a small tolerance to account for numerical imprecision during the computation of the kernel which are
    # then handled by gpytorch during GP computations
    min_tolerated_eigenvalue = -5e-7

    # Vector dimension
    dim_vec = int((dim*dim + dim) / 2)

    # Instantiate the manifold
    spd_manifold = pyman_man.PositiveDefinite(dim)

    # Update the random function of the manifold (the original one samples only eigenvalues between 1 and 2).
    # We need to specify the minimum and maximum eigenvalues of the random matrices.
    spd_manifold.rand = types.MethodType(spd_sample, spd_manifold)
    # Specify the domain
    min_eig = 0.001
    max_eig = 5.
    spd_manifold.min_eig = min_eig
    spd_manifold.max_eig = max_eig

    # Origin in the manifold
    origin_man = symmetric_matrix_to_vector_mandel(np.eye(dim))

    # Define the range of parameter for the kernel
    nb_params = 30
    if dim == 2:
        betas = np.logspace(-1, 2, nb_params)
    elif dim == 3:
        betas = np.logspace(-1.1, 1, nb_params)
    elif dim >= 5:
        betas = np.logspace(-1.5, 0.8, nb_params)

    min_eigval_trials = []

    for trial in range(nb_trials):
        print('Trial ', trial)

        # Sample data
        data_man = np.array([symmetric_matrix_to_vector_mandel(spd_manifold.rand()) for i in range(nb_samples)])
        nb_data = data_man.shape[0]

        # Remove too-ill-conditionned matrices
        id_to_remove = []
        for n in range(nb_data):
            if np.linalg.cond(vector_to_symmetric_matrix_mandel(data_man[n])) > 100:
                id_to_remove.append(n)
        data_man = np.delete(data_man, id_to_remove, axis=0)
        nb_data = data_man.shape[0]

        # Define and compute the kernel for the parameters
        K = []
        min_eigval = []

        for i in range(nb_params):
            # Create kernel instance and set beta
            k = SpdAffineInvariantGaussianKernel(beta_min=0.0)
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
    pd_kernels[pd_kernels > min_tolerated_eigenvalue] = 1.
    pd_kernels[pd_kernels <= min_tolerated_eigenvalue] = 0.
    percentage_pd_kernels = np.sum(pd_kernels, axis=0) / nb_trials

    print(betas)
    print(percentage_pd_kernels)

    # Plot input data if dim is 2
    if dim == 2:
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
        plot_spd_cone(ax, r=10., lim_fact=0.8)

        # Plot training data on the manifold
        plt.plot(data_man[:, 0], data_man[:, 1], data_man[:, 2] / np.sqrt(2), color='k', marker='.', linewidth=0.,
                 markersize=3.)

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
    ax.set_ylabel(r'$PD percentage of $\bm{K}$')
    plt.show()

    # Plot min eigenvalue and percentage of PD kernel in function of the kernel parameter (one graph)
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(np.log10(betas), min_eigval_mean, color='orchid', marker='o')
    ax1.fill_between(np.log10(betas), min_eigval_mean - min_eigval_std, min_eigval_mean + min_eigval_std,
                     color='orchid', alpha=0.2)
    ax2.plot(np.log10(betas), percentage_pd_kernels * 100, color='darkblue', marker='o')
    ax2.plot(np.log10(betas), np.zeros(nb_params), color='k')

    ax1.tick_params(labelsize=30)
    ax2.tick_params(labelsize=30)
    ax1.locator_params(axis='y', nbins=4)
    ax2.locator_params(axis='y', nbins=4)

    ax1.set_xlabel(r'$\log_{10}(\beta)$', fontsize=44)
    ax1.set_ylabel(r'$\lambda_{\min}(\bm{K})$', fontsize=44)
    ax2.set_ylabel(r'PD percentage of $\bm{K}$', fontsize=44)

    ax2.set_ylim(-10., 110)
    align_y_axis(ax1, 0, ax2, 0)

    filename = '../../../Figures/spd' + str(dim) + '_kernel_params.png'
    plt.savefig(filename, bbox_inches='tight')
