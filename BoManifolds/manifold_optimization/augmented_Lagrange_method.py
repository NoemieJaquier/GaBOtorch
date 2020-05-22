import time

import numpy as np

import types

import torch

import pymanopt
from pymanopt.solvers.solver import Solver

from BoManifolds.pymanopt_addons.problem import Problem

from BoManifolds.manifold_optimization.approximate_hessian import get_hessianfd

'''
This file is part of the GaBOtorch library.
Authors: Noemie Jaquier and Leonel Rozo, 2020
License: MIT
Contact: noemie.jaquier@idiap.ch, leonel.rozo@de.bosch.com

This code is an adaptation of the original Matlab code "Optimization-on-manifolds-with-extra-constraints",
(https://github.com/losangle/Optimization-on-manifolds-with-extra-constraints in solvers/almbddmultiplier.m)
This code is based on the paper "Simple algorithms for optimization on Riemannian manifolds with constraints", 
C. Liu and N. Boumal, in Applied Mathematics & Optimization, 2019.
'''


class AugmentedLagrangeMethod(Solver):

    def __init__(self, inner_solver, bound=20, rho_init=1, thetarho=0.3, tau=0.8, starting_tolgradnorm=1e-3,
                 ending_tolgradnorm=1e-6, lambdas_fact=1., gammas_fact=1., *args, **kwargs):
        """
        Augmented Lagrange method, based on "Simple algorithms for optimization on Riemannian manifolds with
        constraints", Liu & Boumal, 2019.

        Parameters
        ----------
        :param inner_solver: solver for the unconstrained subproblem on manifold

        Optional parameters
        -------------------
        :param bound: multipliers (lambdas, gammas) bound
        :param rho_init: initial penalty parameter
        :param thetarho: update factor of the penalty parameter rho
        :param tau: update condition factor for the penalty parameter rho
        :param starting_tolgradnorm: initial accuracy tolerance
        :param ending_tolgradnorm: final accuracy tolerance
        :param lambdas_fact: value of initial Lagrange multipliers for inequality constraints if not specified
        :param gammas_fact: value of initial Lagrange multipliers for equality constraints if not specified
        :param args:
        :param kwargs:
        """
        super(AugmentedLagrangeMethod, self).__init__(*args, **kwargs)

        self.inner_solver = inner_solver
        self._bound = bound
        self._thetarho = thetarho
        self._tau = tau
        self._starting_tolgradnorm = starting_tolgradnorm
        self._ending_tolgradnorm = ending_tolgradnorm
        self._rho_init = rho_init
        self._lambdas_fact = lambdas_fact
        self._gammas_fact = gammas_fact

    def solve(self, problem, x=None, eq_constraints=None, ineq_constraints=None, lambdas=None, gammas=None, rho=None):
        """
        Solve an optimization problem with the augmented Lagrange method.

        Parameters
        ----------
        :param problem: problem to solve (pymanopt format)

        Optional parameters
        -------------------
        :param x: initial point for the optimizer
        :param eq_constraints: equality constraint or list of equality constraints, satisfied if = 0
        :param ineq_constraints: inequality constraint or list of inequality constraints, satisfied if >= 0
        :param lambdas: Lagrange multipliers for inequality constraints
        :param gammas: Lagrange multipliers for equality constraints
        :param rho: penalty parameter

        Returns
        -------
        :return: optimal value on the manifold
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost

        # If no starting point is specified, generate one at random.
        if x is None:
            # Check if the inner solver is a population-based method
            if hasattr(self.inner_solver, '_populationsize'):
                if self.inner_solver._populationsize is not None:
                    x = [man.rand() for i in range(int(self.inner_solver._populationsize))]
                else:
                    x = [man.rand() for i in range(int(min(40, 10 * man.dim)))]
                xbest = x[0]  # take x best as first x

            # Check if the inner solver is Nelder mead
            elif isinstance(self.inner_solver, pymanopt.solvers.NelderMead):
                x = [man.rand() for i in range(int(man.dim + 1))]
                xbest = x[0]  # take x best as first x

            # Otherwise, initialize with one value
            else:
                x = man.rand()
                xbest = x
        # elif isinstance(x, list):  # Removed for product of manifolds, not sure why this was originally implemented
        #     xbest = x[0]
        else:
            xbest = x

        xbest_prev = xbest

        # Set constraints
        if eq_constraints is None:
            eq_constraints = []
        if not isinstance(eq_constraints, list):
            eq_constraints = [eq_constraints]

        if ineq_constraints is None:
            ineq_constraints = []
        if not isinstance(ineq_constraints, list):
            ineq_constraints = [ineq_constraints]

        neq_cons = len(eq_constraints)
        nineq_cons = len(ineq_constraints)

        # Create problems for constraints
        eq_con_probs = [Problem(problem.manifold, eq_constraints[i], arg=torch.Tensor(), verbosity=verbosity)
                        for i in range(neq_cons)]
        ineq_con_probs = [Problem(problem.manifold, ineq_constraints[i], arg=torch.Tensor(), verbosity=verbosity)
                          for i in range(nineq_cons)]

        # Set multipliers
        if lambdas is None and nineq_cons > 0:
            lambdas = self._lambdas_fact * np.ones(nineq_cons)

        if gammas is None and neq_cons > 0:
            gammas = self._gammas_fact *np.ones(neq_cons)

        if rho is None:
            rho = self._rho_init

        # Set parameters
        oldacc = np.inf
        tolgradnorm = self._starting_tolgradnorm
        theta_tolgradnorm = (self._ending_tolgradnorm / self._starting_tolgradnorm) ** (1. / self._maxiter)

        # Initializations
        time0 = time.time()

        # k counts the outer iterations. The semantic is that k counts the number of iterations fully executed so far.
        k = 0

        self._start_optlog()

        while True:
            # ************************
            # ** Solving Subproblem **
            # ************************
            # Create subproblem
            subproblem = self.subproblem_alm(problem, eq_con_probs, ineq_con_probs, lambdas, gammas, rho)

            # Solve it
            self.inner_solver._mingradnorm = tolgradnorm
            xbest = self.inner_solver.solve(subproblem, x)

            # Update the multipliers
            newacc = 0.
            for con in range(nineq_cons):
                cost_iter = ineq_con_probs[con].cost(xbest)
                newacc = max(newacc, abs(max(-lambdas[con]/rho, cost_iter)))
                lambdas[con] = min(self._bound, max(lambdas[con] + rho * cost_iter, 0.))

            for con in range(neq_cons):
                cost_iter = eq_con_probs[con].cost(xbest)
                newacc = max(newacc, abs(cost_iter))
                gammas[con] = min(self._bound, max(-self._bound, gammas[con] + rho * cost_iter))

            # Update the penalty parameter rho
            if k == 0 or newacc > self._tau * oldacc:
                rho = rho/self._thetarho
            oldacc = newacc

            # Update tolerance
            tolgradnorm = max(self._ending_tolgradnorm, tolgradnorm * theta_tolgradnorm)

            # k is the number of iterations we have accomplished.
            k = k + 1

            # ** Display:
            if verbosity >= 2:
                print("%5d\t%+.16e" % (k, objective(xbest)))

            # ** CHECK STOPPING criteria

            stop_reason = self._check_stopping_criterion(time0, iter=k, stepsize=man.dist(xbest, xbest_prev))
            if tolgradnorm <= self._ending_tolgradnorm:
                stop_reason = ("Terminated - min grad norm reached after %d iterations, %.2f seconds."
                               % (k, (time.time() - time0)))

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

            # Update initial x
            # Check if the inner solver is a population-based method
            if hasattr(self.inner_solver, '_populationsize'):
                x = [man.rand() for i in range(int(self.inner_solver._populationsize))]
                # Best value as first sample
                x[0] = xbest
            # Check if the inner solver is Nelder mead
            elif isinstance(self.inner_solver, pymanopt.solvers.NelderMead):
                x = [man.rand() for i in range(int(man.dim + 1))]
                # Best value as first sample
                x[0] = xbest
            # Otherwise, initialize with the best value
            else:
                x = xbest

            # Update previous xbest
            xbest_prev = xbest

        if self._logverbosity <= 0:
            return xbest
        else:
            self._stop_optlog(xbest, objective(xbest), stop_reason, time0, iter=iter)
            return xbest, self._optlog

    def subproblem_alm(self, problem, eq_con_probs=None, ineq_con_probs=None, lambdas=None, gammas=None, rho=1.):
        """
        Define the unconstrainted subproblem on the manifold for the augmented Lagrange method.

        Parameters
        ----------
        :param problem: original problem (pymanopt format)

        Optional parameters
        -------------------
        :param eq_con_probs: equality constraint problem or list of equality constraints problems (pymanopt format),
            satisfied if = 0
        :param ineq_con_probs: inequality constraint problem or list of inequality constraints problems (pymanopt
            format), satisfied if >= 0
        :param lambdas: Lagrange multipliers for inequality constraints
        :param gammas: Lagrange multipliers for equality constraints
        :param rho: penalty parameter

        Returns
        -------
        :return: unconstrained subproblem (pymanopt format)
        """

        # Set constraints problems
        if eq_con_probs is None:
            eq_con_probs = []
        if not isinstance(eq_con_probs, list):
            eq_con_probs = [eq_con_probs]

        if ineq_con_probs is None:
            ineq_con_probs = []
        if not isinstance(ineq_con_probs, list):
            ineq_con_probs = [ineq_con_probs]

        neq_cons = len(eq_con_probs)
        nineq_cons = len(ineq_con_probs)

        def subproblem_cost(self, x):
            cost = problem.cost(x)

            # Add inequality constraints costs
            if nineq_cons > 0:
                for con in range(nineq_cons):
                    # Here we have - ineq_con_probs, because in our case the constraints is satisfied if g(x)>=0
                    cost += rho / 2. * max(0., (lambdas[con] / rho - ineq_con_probs[con].cost(x))) ** 2

            # Add equality constraints costs
            if neq_cons > 0:
                for con in range(neq_cons):
                    cost += rho / 2 * (gammas[con] / rho + eq_con_probs[con].cost(x)) ** 2

            return cost

        def subproblem_grad(self, x):
            grad = problem.grad(x)

            # Add inequality constraints costs
            if nineq_cons > 0:
                for con in range(nineq_cons):
                    # Here we have minus signs because in our case the constraints is satisfied if g(x)>=0
                    tmp_con_cost = ineq_con_probs[con].cost(x)
                    if lambdas[con] / rho - tmp_con_cost > 0:
                        if type(x) in (list, tuple) or issubclass(type(x), (list, tuple)):
                            # Handle the case where x is a list or a tuple (typically for products of manifolds
                            ineq_con_grad = ineq_con_probs[con].grad(x)
                            for k in range(len(x)):
                                grad[k] += (tmp_con_cost * rho - lambdas[con]) * ineq_con_grad[k]
                        else:
                            grad += (tmp_con_cost * rho - lambdas[con]) * ineq_con_probs[con].grad(x)

            # Add equality constraints costs
            if neq_cons > 0:
                for con in range(neq_cons):
                    if type(x) in (list, tuple) or issubclass(type(x), (list, tuple)):
                        # Handle the case where x is a list or a tuple (typically for products of manifolds
                        eq_con_cost = eq_con_probs[con].cost(x)
                        eq_con_grad = eq_con_probs[con].grad(x)
                        for k in range(len(x)):
                            grad[k] += (eq_con_cost * rho + gammas[con]) * eq_con_grad[k]
                    else:
                        grad += (eq_con_probs[con].cost(x) * rho + gammas[con]) * eq_con_probs[con].grad(x)

            return grad

        # Subproblem definition
        subproblem = Problem(problem.manifold, cost=subproblem_cost, verbosity=problem.verbosity)

        # We redefine the methods cost, grad and hessian.
        # This is because we do not want pymanopt to use a backend here as all the derivative methods are specified.
        subproblem._cost = types.MethodType(subproblem_cost, problem)
        subproblem._grad = types.MethodType(subproblem_grad, problem)
        subproblem._hess = types.MethodType(get_hessianfd, problem)

        return subproblem



