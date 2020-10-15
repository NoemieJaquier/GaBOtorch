from collections import OrderedDict
from math import inf
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.nn import Module

from botorch.optim.numpy_converter import ParameterBounds, TorchAttr

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the functions of botorch (in botorch.optim.numpy_converter.py).
'''


def module_to_list_of_array(
    module: Module,
    bounds: Optional[ParameterBounds] = None,
    exclude: Optional[Set[str]] = None,
) -> Tuple[list, Dict[str, TorchAttr], Optional[list]]:
    """
    This function extract named parameters from a module into a list of numpy arrays. It only extracts parameters with
    requires_grad, since it is meant for optimizing.

    Parameters
    ----------
    :param module: A module with parameters. May specify parameter constraints in a `named_parameters_and_constraints`
        method.

    Optional parameters
    -------------------
    :param bounds: A ParameterBounds dictionary mapping parameter names to tuples of lower and upper bounds.
        Bounds specified here take precedence over bounds on the same parameters specified in the constraints
        registered with the module.
    :param exclude: A list of parameter names that are to be excluded from extraction.

    Returns
    -------
    :return: 3-element tuple containing
        - The parameter values as a list of numpy arrays.
        - An ordered dictionary with the name and tensor attributes of each parameter.
        - A list of `2 x n_params` numpy array with lower and upper bounds if at least one constraint is finite, and
            None otherwise.

    Example:
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        parameter_array, property_dict, bounds_out = module_to_array(mll)
    """
    x: List[np.ndarray] = []
    lower: List[np.ndarray] = []
    upper: List[np.ndarray] = []
    property_dict = OrderedDict()
    exclude = set() if exclude is None else exclude

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(module, "named_parameters_and_constraints"):
        for param_name, _, constraint in module.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    for p_name, t in module.named_parameters():
        if p_name not in exclude and t.requires_grad:
            property_dict[p_name] = TorchAttr(
                shape=t.shape, dtype=t.dtype, device=t.device
            )
            if t.ndim > 1 and t.shape[0] > 1:  # if the variable is a matrix, keep its shape
                x.append(t.detach().cpu().double().clone().numpy())
            else:  # Vector case
                x.append(t.detach().view(-1).cpu().double().clone().numpy())
            # construct bounds
            if bounds_:
                l_, u_ = bounds_.get(p_name, (-inf, inf))
                if torch.is_tensor(l_):
                    l_ = l_.cpu().detach()
                if torch.is_tensor(u_):
                    u_ = u_.cpu().detach()
                # check for Nones here b/c it may be passed in manually in bounds
                lower.append(np.full(t.nelement(), l_ if l_ is not None else -inf))
                upper.append(np.full(t.nelement(), u_ if u_ is not None else inf))

    return x, property_dict, bounds


def set_params_with_list_of_array(
    module: Module, x: list, property_dict: Dict[str, TorchAttr]
) -> Module:
    """
    This function sets module parameters with values from numpy array.

    Parameters
    ----------
    :param module: a module with parameters to be set
    :param x: the numpy array containing parameter values
    :param property_dict: a dictionary of parameter names and torch attributes as returned by module_to_array.

    Returns
    -------
    :return: a module with parameters updated in-place.

    Example:
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    parameter_array, property_dict, bounds_out = module_to_array(mll)
    parameter_array += 0.1  # perturb parameters (for example only)
    mll = set_params_with_array(mll, parameter_array,  property_dict)
    """
    param_dict = OrderedDict(module.named_parameters())
    idx = 0
    for p_name, attrs in property_dict.items():
        # Construct the new tensor
        if len(attrs.shape) == 0:  # deal with scalar tensors
            new_data = torch.tensor(x[idx][0], dtype=attrs.dtype, device=attrs.device)
        else:
            new_data = torch.tensor(x[idx], dtype=attrs.dtype, device=attrs.device).view(*attrs.shape)
        idx += 1
        # Update corresponding parameter in-place. Disable autograd to update.
        param_dict[p_name].requires_grad_(False)
        param_dict[p_name].copy_(new_data)
        param_dict[p_name].requires_grad_(True)
    return module
