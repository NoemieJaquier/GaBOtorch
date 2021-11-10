# GaBOtorch
This repository contains the source code to perform Geometry-aware Bayesian Optimization (GaBO) and High-Dimensional Geometry-aware Bayesian Optimization (HD-GaBO) on Riemannian manifolds.

# Dependencies
This code runs with Python>=3.6. It requires the following packages:
- numpy
- scipy
- matplotlib
- pymanopt
- torch
- gpytorch
- botorch

# Installation 
To install GaBOtorch, first clone the repository and install the related packages, as explained below.

```
pip install numpy scipy matplotlib pymanopt torch gpytorch botorch
```
Finally, from the GaBOtorch folder, run
```
pip install -e .
```


# Examples
The following examples are available in GaBOtorch:
### Kernels [1]
| Sphere manifold      |           | 
|:------------- |:-------------| 
| sphere_kernels      | This example shows the use of different kernels for the hypershere manifold S^n , used for Gaussian process regression. | 
| sphere_gaussian_kernel_parameters      | This example shows the experimental selection of parameters for the Sphere Gaussian kernel.      |

| SPD manifold       |           | 
|:------------- |:-------------| 
| spd_kernels      | This example shows the use of different kernels for the SPD manifold, used for Gaussian process regression | 
| spd_gaussian_kernel_parameters      | This example shows the experimental selection of parameters for the SPD Affine-Invariant Gaussian kernel.  |


### BO on the sphere [1]
| Benchmark examples      |           | 
|:------------- |:-------------| 
| gabo_sphere      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the sphere S2 to optimize the Ackley function. | 
| bo_euclidean_sphere      | This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function.  |

| Constrained benchmark examples      |           | 
|:------------- |:-------------| 
| gabo_sphere_bound_constraints      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the sphere S2 to optimize the Ackley function. In this example, the search domain is bounded and represents a subspace of the manifold. | 
| gabo_sphere_equality_constraints    | This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function.  In this example, the parameters must satisfy equality constraints and the search space represents a subspace of the manifold. |
| gabo_sphere_equality_constraints    | This example shows the use of Euclidean Bayesian optimization on the sphere S2 to optimize the Ackley function.  In this example, the parameters must satisfy inequality constraints and the search space represents a subspace of the manifold. |

### BO on the SPD manifold [1]
| Benchmark examples      |           | 
|:------------- |:-------------| 
| gabo_spd      | This example shows the use of Geometry-aware Bayesian optimization (GaBO) on the SPD manifold S2_++ to optimize the Ackley function. | 
| bo_cholesky_spd      | This example shows the use of Cholesky Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function. An Euclidean BO is applied on the Cholesky decomposition of the SPD matrices.  | 
| bo_euclidean_spd      | This example shows the use of Euclidean Bayesian optimization on the SPD manifold S2_++ to optimize the Ackley function. |

### High-dimentional BO on the sphere [2]
| Benchmark examples      |           | 
|:------------- |:-------------| 
| hd_gabo_sphere      | This example shows the use of High-Dimensional Geometry-aware Bayesian optimization (HD-GaBO) to optimize the Ackley function on a latent sphere S2 embedded in the sphere S5. | 

### High-diemnsional BO on the SPD manifold [2]
| Benchmark examples      |           | 
|:------------- |:-------------| 
| hd_gabo_spd      | This example shows the use of High-Dimensional Geometry-aware Bayesian optimization (HD-GaBO)  to optimize the Rosenbrock function on a latent SPD manifold S2_++ embedded in the SPD manifold S5_++. | 

# References
If you found GaBOtorch useful, we would be grateful if you cite the following references:

[[1](http://njaquier.ch/files/CoRL19_Jaquier_GaBO.pdf)] N. Jaquier, L. Rozo, S. Calinon, and M. Bürger (2019). Bayesian Optimization meets Riemannian Manifolds in Robot Learning. In Conference on Robot Learning (CoRL).

[[2](http://njaquier.ch/files/HDGaBO.pdf)] N. Jaquier, and L. Rozo (2020). High-dimensional Bayesian Optimization via Nested Riemannian Manifolds. In Neural Information Processing Systems (NeurIPS). 

```
@inproceedings{Jaquier19GaBO,
	author="Jaquier, N and Rozo, L. and Calinon, S. and B\"urger, M.", 
	title="Bayesian Optimization meets Riemannian Manifolds in Robot Learning",
	booktitle="In Conference on Robot Learning (CoRL)",
	year="2019",
	pages=""
}
@inproceedings{Jaquier20HDGaBO,
	author="Jaquier, N and Rozo, L.", 
	title="High-dimensional Bayesian Optimization via Nested Riemannian Manifolds",
	booktitle="Neural Information Processing Systems (NeurIPS)",
	year="2020",
	pages=""
}
```
