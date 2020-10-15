import time
import types
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union
from operator import attrgetter
from collections import OrderedDict

import numpy as np
import torch
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from torch.nn import Module

from botorch.optim.utils import (
    _get_extra_mll_args,
)

from pymanopt.manifolds import Euclidean, Product
from pymanopt.solvers.solver import Solver
import pymanopt.solvers as pyman_solvers

from BoManifolds.manifold_optimization.numpy_list_converter import TorchAttr, module_to_list_of_array, \
    set_params_with_list_of_array
from BoManifolds.manifold_optimization.approximate_hessian import get_hessianfd

from BoManifolds.pymanopt_addons.problem import Problem

ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [np.ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, np.ndarray]
]
TModToArray = Callable[
    [Module, Optional[ParameterBounds], Optional[Set[str]]],
    Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]],
]
TArrayToMod = Callable[[Module, np.ndarray, Dict[str, TorchAttr]], Module]

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the functions of botorch (in botorch.fit).
'''


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


def fit_gpytorch_manifold(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    solver: Solver = pyman_solvers.ConjugateGradient(maxiter=500),
    nb_init_candidates: int = 200,
    last_x_as_candidate_prob: float = 0.9,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = False,
    module_to_array_func: TModToArray = module_to_list_of_array,
    module_from_array_func: TArrayToMod = set_params_with_list_of_array,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    """
    This function fits a gpytorch model by maximizing MLL with a pymanopt optimizer.

    The model and likelihood in mll must already be in train mode.
    This method requires that the model has `train_inputs` and `train_targets`.

    Parameters
    ----------
    :param mll: MarginalLogLikelihood to be maximized.

    Optional parameters
    -------------------
    :param nb_init_candidates: number of random initial candidates for the GP parameters
    :param last_x_as_candidate_prob: probability that the last set of parameter is among the initial candidates
    :param bounds: A dictionary mapping parameter names to tuples of lower and upper bounds.
    :param solver: Pymanopt solver.
    :param options: Dictionary of solver options, passed along to scipy.minimize.
    :param track_iterations: Track the function values and wall time for each iteration.
    :param approx_mll: If True, use gpytorch's approximate MLL computation. This is disabled by default since the
        stochasticity is an issue for determistic optimizers). Enabling this is only recommended when working with
        large training data sets (n>2000).

    Returns
    -------
    :return: 2-element tuple containing
        - MarginalLogLikelihood with parameters optimized in-place.
        - Dictionary with the following key/values:
            "fopt": Best mll value.
            "wall_time": Wall time of fitting.
            "iterations": List of OptimizationIteration objects with information on each iteration.
                If track_iterations is False, will be empty.

    Example:
    gp = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.train()
    fit_gpytorch_scipy(mll)
    mll.eval()
    """
    options = options or {}
    # Current parameters
    x0, property_dict, bounds = module_to_array_func(module=mll, bounds=bounds, exclude=options.pop("exclude", None))
    x0 = [x0i.astype(np.float64) for x0i in x0]
    if bounds is not None:
        warnings.warn('Bounds handling not supported yet in fit_gpytorch_manifold')
        # bounds = Bounds(lb=bounds[0], ub=bounds[1], keep_feasible=True)

    t1 = time.time()

    # Define cost function
    def cost(x):
        param_dict = OrderedDict(mll.named_parameters())
        idx = 0
        for p_name, attrs in property_dict.items():
            # Construct the new tensor
            if len(attrs.shape) == 0:  # deal with scalar tensors
                # new_data = torch.tensor(x[0], dtype=attrs.dtype, device=attrs.device)
                new_data = torch.tensor(x[idx][0], dtype=attrs.dtype, device=attrs.device)
            else:
                # new_data = torch.tensor(x, dtype=attrs.dtype, device=attrs.device).view(*attrs.shape)
                new_data = torch.tensor(x[idx], dtype=attrs.dtype, device=attrs.device).view(*attrs.shape)
            param_dict[p_name].data = new_data
            idx += 1
        # mllx = set_params_with_array(mll, x, property_dict)
        train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
        mll.zero_grad()
        output = mll.model(*train_inputs)
        args = [output, train_targets] + _get_extra_mll_args(mll)
        loss = -mll(*args).sum()
        return loss

    def egrad(x):
        loss = cost(x)
        loss.backward()
        param_dict = OrderedDict(mll.named_parameters())
        grad = []
        for p_name in property_dict:
            t = param_dict[p_name].grad
            if t is None:
                # this deals with parameters that do not affect the loss
                if len(property_dict[p_name].shape) > 1 and property_dict[p_name].shape[0] > 1:
                    # if the variable is a matrix, keep its shape
                    grad.append(np.zeros(property_dict[p_name].shape))
                else:
                    grad.append(np.zeros(property_dict[p_name].shape))
            else:
                if t.ndim > 1 and t.shape[0] > 1:  # if the variable is a matrix, keep its shape
                    grad.append(t.detach().cpu().double().clone().numpy())
                else:  # Vector case
                    grad.append(t.detach().view(-1).cpu().double().clone().numpy())
        return grad

    # Define the manifold (product of manifolds)
    manifolds_list = []
    for p_name, t in mll.named_parameters():
        try:
            # If a manifold is given add it
            manifolds_list.append(attrgetter(p_name + "_manifold")(mll))
        except AttributeError:
            # Otherwise, default: Euclidean
            manifolds_list.append(Euclidean(int(np.prod(property_dict[p_name].shape))))
    # Product of manifolds
    manifold = Product(manifolds_list)

    # Instanciate the problem on the manifold
    if track_iterations:
        verbosity = 2
    else:
        verbosity = 0

    problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=verbosity, arg=torch.Tensor()) #, precon=precon)

    # For cases where the Hessian is hard/long to compute, we approximate it with finite differences of the gradient.
    # Typical cases: the Hessian can be hard to compute due to the 2nd derivative of the eigenvalue decomposition,
    # e.g. in the SPD affine-invariant distance.
    problem._hess = types.MethodType(get_hessianfd, problem)

    # Choose initial parameters
    # Do not always consider x0, to encourage variations of the parameters.
    if np.random.rand() < last_x_as_candidate_prob:
        x0_candidates = [x0]
        x0_candidates += [manifold.rand() for i in range(nb_init_candidates - 1)]
    else:
        x0_candidates = []
        x0_candidates += [manifold.rand() for i in range(nb_init_candidates)]
    for i in range(int(3*nb_init_candidates/4)):
        x0_candidates[i][0:4] = x0[0:4]  #TODO remove hard-coding
    y0_candidates = [cost(x0_candidates[i]) for i in range(nb_init_candidates)]

    y_init, x_init_idx = torch.Tensor(y0_candidates).min(0)
    x_init = x0_candidates[x_init_idx]

    with gpt_settings.fast_computations(log_prob=approx_mll):
        # Logverbosity of the solver to 1
        solver._logverbosity = 1
        # Solve
        opt_x, opt_log = solver.solve(problem, x=x_init)

    # Construct info dict
    info_dict = {
        "fopt": float(cost(opt_x).detach().numpy()),
        "wall_time": time.time() - t1,
        "opt_log": opt_log,
    }
    # if not res.success:  # TODO update
    #     try:
    #         # Some res.message are bytes
    #         msg = res.message.decode("ascii")
    #     except AttributeError:
    #         # Others are str
    #         msg = res.message
    #     warnings.warn(
    #         f"Fitting failed with the optimizer reporting '{msg}'", OptimizationWarning
    #     )
    # Set to optimum
    mll = module_from_array_func(mll, opt_x, property_dict)
    return mll, info_dict
