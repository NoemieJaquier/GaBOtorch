import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from scipy.optimize import minimize, Bounds

import torch
from torch import Tensor
from torch.nn import Module

from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.gen import get_best_candidates
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.initializers import initialize_q_batch, initialize_q_batch_nonneg
from botorch.optim.utils import columnwise_clamp, fix_features

from botorch.optim.parameter_constraints import (
    _arrayify,
    make_scipy_bounds,
)
'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


# This function is based on (and very similar to) the botorch.optim.joint_optimize function of botorch.
def joint_optimize(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    constraints = (),
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_init: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    This function generates a set of candidates via joint multi-start optimization

    Parameters
    ----------
    :param acq_function: the acquisition function
    :param bounds: a `2 x d` tensor of lower and upper bounds for each column of `X`
    :param q: number of candidates
    :param num_restarts: number of starting points for multistart acquisition function optimization
    :param raw_samples: number of samples for initialization


    Optional parameters
    -------------------
    :param options: options for candidate generation
    :param constraints: constraints in scipy format
    :param fixed_features: A map {feature_index: value} for features that should be fixed to a particular value
        during generation.
    :param post_processing_init: A function that post processes the generated initial samples
        (e.g. so that they fulfill some constraints).

    Returns
    -------
    :return: a `q x d` tensor of generated candidates.
    """

    options = options or {}
    batch_initial_conditions = \
        gen_batch_initial_conditions(acq_function=acq_function, bounds=bounds,
                                     q=None if isinstance(acq_function, AnalyticAcquisitionFunction) else q,
                                     num_restarts=num_restarts, raw_samples=raw_samples,
                                     options=options, post_processing_init=post_processing_init)
    batch_limit = options.get("batch_limit", num_restarts)
    batch_candidates_list = []
    batch_acq_values_list = []
    start_idx = 0
    while start_idx < num_restarts:
        end_idx = min(start_idx + batch_limit, num_restarts)
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = \
            gen_candidates_scipy(initial_conditions=batch_initial_conditions[start_idx:end_idx],
                                 acquisition_function=acq_function, lower_bounds=bounds[0], upper_bounds=bounds[1],
                                 options={k: v for k, v in options.items() if k not in ("batch_limit", "nonnegative")},
                                 constraints=constraints, fixed_features=fixed_features)
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        start_idx += batch_limit

    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)
    return get_best_candidates(batch_candidates=batch_candidates, batch_values=batch_acq_values)


# This function is based on (and very similar to) the botorch.gen.gen_candidates_scipy function of botorch.
def gen_candidates_scipy(
    initial_conditions: Tensor,
    acquisition_function: Module,
    lower_bounds: Optional[Union[float, Tensor]] = None,
    upper_bounds: Optional[Union[float, Tensor]] = None,
    constraints=(),
    options: Optional[Dict[str, Any]] = None,
    fixed_features: Optional[Dict[int, Optional[float]]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    This function generates a set of candidates using `scipy.optimize.minimize`

    Parameters
    ----------
    :param initial_conditions: starting points for optimization
    :param acquisition_function: acquisition function to be optimized

    Optional parameters
    -------------------
    :param lower_bounds: minimum values for each column of initial_conditions
    :param upper_bounds: maximum values for each column of initial_conditions
    :param constraints: constraints in scipy format
    :param options: options for candidate generation
    :param fixed_features: A map {feature_index: value} for features that should be fixed to a particular value
        during generation.

    Returns
    -------
    :return: 2-element tuple containing the set of generated candidates and the acquisition value for each t-batch.
    """

    options = options or {}
    x0 = columnwise_clamp(initial_conditions, lower_bounds, upper_bounds).requires_grad_(True)

    bounds = Bounds(lb=lower_bounds, ub=upper_bounds, keep_feasible=True)

    def f(x):
        X = (torch.from_numpy(x).to(initial_conditions).contiguous().requires_grad_(True))
        X_fix = fix_features(X=X, fixed_features=fixed_features)
        loss = -acquisition_function(X_fix[None]).sum()
        # compute gradient w.r.t. the inputs (does not accumulate in leaves)
        gradf = _arrayify(torch.autograd.grad(loss, X)[0].contiguous().view(-1))
        fval = loss.item()
        return fval, gradf

    candidates = torch.zeros(x0.shape, dtype=torch.float64)
    # TODO this does not handle the case where q!=1
    for i in range(x0.shape[0]):
        res = minimize(f, x0[i, 0].detach().numpy(), method="SLSQP", jac=True, bounds=bounds, constraints=constraints,
                       options={k: v for k, v in options.items() if k != "method"},)
        candidates[i] = fix_features(X=torch.from_numpy(res.x).to(initial_conditions).contiguous(),
                                     fixed_features=fixed_features,)

    batch_acquisition = acquisition_function(candidates)

    return candidates, batch_acquisition


# This function is based on (and very similar to) the botorch.optim.gen_batch_initial_conditions function of botorch.
def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    post_processing_init: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """
    This function generates a batch of initial conditions for random-restart optimization

    Parameters
    ----------
    :param acq_function: the acquisition function to be optimized.
    :param bounds: a `2 x d` tensor of lower and upper bounds for each column of `X`
    :param q: number of candidates
    :param num_restarts: number of starting points for multistart acquisition function optimization
    :param raw_samples: number of samples for initialization

    Optional parameters
    -------------------
    :param options: options for candidate generation
    :param post_processing_init: A function that post processes the generated initial samples
        (e.g. so that they fulfill some constraints).

    Returns
    -------
    :return: a `num_restarts x q x d` tensor of initial conditions
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")  # pyre-ignore
    batch_limit: Optional[int] = options.get("batch_limit")  # pyre-ignore
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            X_rnd = draw_sobol_samples(bounds=bounds, n=raw_samples * factor, q=1 if q is None else q, seed=seed,)

            # Constraints the samples
            if post_processing_init is not None:
                X_rnd = post_processing_init(X_rnd)

            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]

                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(X_rnd[start_idx:end_idx])
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list).to(X_rnd)

            batch_initial_conditions = init_func(X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs)

            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions
