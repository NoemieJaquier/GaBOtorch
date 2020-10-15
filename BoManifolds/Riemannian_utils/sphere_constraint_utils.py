import numpy as np
'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com

The functions of this file are based on the function of botorch (in botorch.optim).
'''


def norm_one_constraint(x):
    """
    This function defines an 1-norm equality constraint on the a vector.
    The value returned by the function is 0 if the equality constraints is satisfied.

    Parameters
    ----------
    :param x: vector

    Returns
    -------
    :return: difference between the norm of x and 1
    """
    return np.linalg.norm(x) - 1.
