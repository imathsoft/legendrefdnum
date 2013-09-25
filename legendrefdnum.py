# -*- coding: utf-8 -*-
#
# legendrefdnum, a numerical FD-method solver for Sturm-Liouville problems
# Copyright (C) 2013, Danyil Bohdan, Denys Dragunov
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

"""A numerical FD-method solver for Sturm-Liouville problems.

The below code references formulas found in the article Volodymyr L. Makarov,
Denys V. Dragunov, Danyil V. Bohdan. Exponentially convergent
numerical-analytical method for solving eigenvalue problems for singular
differential operators. A preprint of this article will soon be availible from
http://arxiv.org/. It will be referred to as [1] in the comments below.

It is highly recommended that you use this module with
numpy.longdouble == float128. Doing so requires running a 64-bit version of
NumPy and a 64-bit operating system. The module has been tested on
Ubuntu 12.04 x86_64 with Python 2.7.3.
"""

from __future__ import division
from __future__ import print_function

__author__ = 'Danyil Bohdan, Denys Dragunov'
__copyright__ = 'Copyright (C) 2013, Danyil Bohdan, Denys Dragunov'
__license__ = 'LGPLv2.1+'
__version__ = '1.0.0'

import itertools
import time
import csv
import logging

import numpy
import matplotlib
import matplotlib.pyplot as plt

import scipy.special

from legendrepqwrappers import LegendreP, LegendreQ
from legendrepqwrappers import DLegendreP, DLegendreQ


class DeltaInvFile(object):
    """Provides access to precalculated values of \delta^{-1}.

    See formula (5.2) of [1].

    Use example:
    deltainv = DeltaInvFile(100)
    d1 = deltainv[1]
    """
    def __init__(self, nvalues, filename = "deltainvs.txt",
                  numclass = numpy.longdouble):
        """Initalize and load a deltainv file into memory.

        nvalues is the total number of \delta^{-1} values to load from the
        file and store in memory. filename indicates the where we should load
        those values from.

        A deltainvs file is a text file of the following format:

        <Total number of lines><newline>
        <delta^{-1}_{0} as 0.1234...><newline>
        <delta^{-1}_{1}  as 5.6789...><newline>
        ...
        """
        try:
            with open(filename, "r") as deltainvfile:
                lines = deltainvfile.readlines()
                self.maxn = min(int(lines[0]), nvalues + 1)
                self.deltainvs = [numclass(0.5)] +\
                                 [numclass(x) for x in lines[1:self.maxn]]
        except IOError:
            self.maxn = 0
            self.deltainvs = []

    def __getitem__(self, key):
        return self.deltainvs[key]

    def __len__(self):
        return self.maxn


class IntegratorSinc(object):
    """Integrates functions using the tanh rule and Stenger's formula."""

    def __init__(self, a = -1, b = 1, K = 10, N = 1,
                  numclass = numpy.longdouble, splitpoints = None):
        """Initialize and precalculate the values needed for integration.

        (a, b) is the interval over which integration will be performed.
        K determines the number of points of the mesh.
        N is the number of subintervals (a, b) is split into.

        2 * K + 1 points are generated on each subinterval for a total of
        (2 * K + 1) * N. If splitpoints is None and N > 1 then the interval
        (a, b) is split uniformly into N subintervals else it is split along
        the first N points listed in splitpoints.
        """
        self.a = a
        self.b = b
        self.K = K
        self.N = N
        self.numclass = numclass

        # See formula (5.2) of [1]
        self.hsinc = numpy.sqrt(self.numclass(2) *
                                self.numclass(numpy.pi) / self.numclass(K))

        if splitpoints is not None:
            self.split = numpy.array(splitpoints, dtype=self.numclass)
        else:
            self.split = numpy.zeros((N + 1), dtype=self.numclass)
            for i in range(0, N + 1):
                self.split[i] = self.numclass(a) + self.numclass(b - a) *\
                                self.numclass(i) / self.numclass(N)
        logger.debug("%s", repr(self.split))

        # See formula (5.3) of [1]
        self.z = numpy.zeros((N + 1, 2 * self.K + 1), dtype=self.numclass)
        for i in range(1, N + 1):
            for j in range(-K, K + 1):
                exphk = numpy.exp(self.hsinc * j)
                aa = (self.split[i - 1] + self.split[i] * exphk)
                bb = (1 + exphk)
                self.z[i, K + j] = aa / bb
        logger.debug("\nz: %s", str(self.z))

        # See formula (5.3) of [1]
        self.mu = numpy.zeros((N + 1, 2 * self.K+ 1), dtype=self.numclass)
        for i in range(1, N + 1):
            for j in range(0, K + 1):
                q = numpy.exp(j * self.hsinc / 2)
                self.mu[i, K + j] = (self.split[i] - self.split[i - 1]) /\
                                    (1 / q + q) ** 2
                self.mu[i, K - j] = self.mu[i, K + j]
        logger.debug("\nmu: %s", str(self.mu))

        # Try to load precalculated \delta^{-1}_{i} values for i >= 0.
        # If we can't or there isn't enough of them in the precalc file we
        # generate them with scipy.special.sici instead (not recommended).
        deltainvfile = DeltaInvFile(2 * self.K, numclass = self.numclass)
        self.deltainv = numpy.zeros((4 * self.K + 1), dtype=self.numclass)
        if 2 * self.K <= len(deltainvfile):
            self.deltainv[2 * self.K:] = deltainvfile
        else:
            self.deltainv[0] = 0.5
            for k in range(1, 2 * self.K + 1):
                # the integral of sin(pi * x) / (pi * x) dx from 0 to k
                self.deltainv[2 * self.K + k] = 0.5 +\
                                                scipy.special.sici(numpy.pi *\
                                                k)[0] / numpy.pi
        # \delta^{-1}_{-i} = 1 - \delta^{-1}_{i}
        for k in range(1, 2 * self.K + 1):
            self.deltainv[2 * self.K - k] = 1 - self.deltainv[2 * self.K + k]

    def integr_ab(self, *vals):
        """Integrate vals (lists of values on self.z or functions) over (a, b).

        Every v in vals can be either a list of values or a callable. A list
        of values is interpreted as values of some function on our
        intergrator's mesh, self.z.

        The returned value is a list of values of the integral for each
        subinterval of (a, b).

        Use example:
        intsinc = IntegratorSinc(a = -1, b = 1, K = 2, N = 1)
        # Now intsinc.z ==
        # [-0.94387775 -0.70952513  0.0  0.70952513  0.94387775].
        # Integrate f(x) over (a, b) where f(z[0]) = 0.2, f(z[1]) = 0.4, ...
        v1 = intsinc.integr_ab([0.2, 0.4, 0.6, 0.8, 1.0])
        # Integrate f(x) * g(x) over (a, b) where f(x) = x, g(x) = 1 - x
        v2 = intsinc.integr_ab(lambda x: x, lambda x: 1 - x)
        """
        alist = []
        # For every subinterval of (a, b)...
        for i in range(1, self.N + 1):
            acc = 0.0 # Integral over current subinterval.
             # For each point on the subinterval...
            for j in range(-self.K, self.K + 1):
                # mult accumulates the product of all vals on z_j
                mult = self.mu[i, self.K + j]
                for val in vals:
                    if hasattr(val, '__call__'): # function
                        mult *= val(self.z[i, self.K + j])
                    else: # assume list / array
                        mult *= val[i, self.K + j]
                acc += mult
            alist.append(acc * self.hsinc)
        return numpy.array(alist)

    def integr_az(self, k, *vals):
        """Integrate vals over (a, z_k).

        See the docstring for integr_ab for more information on use."""
        alist = []
        # For every subinterval of (a, b)...
        for i in range(1, self.N + 1):
            acc = 0.0 # Integral over current subinterval.
            # For each point on the subinterval...
            for j in range(-self.K, self.K + 1):
                # mult accumulates the product of all vals on z_j
                mult = self.mu[i, self.K + j] *\
                       self.deltainv[2 * self.K + k - j]
                for val in vals:
                    if hasattr(val, '__call__'): # function
                        mult *= val(self.z[i, self.K + j])
                    else: # assume list / array
                        mult *= val[i, self.K + j]
                acc += mult
            alist.append(acc * self.hsinc)
        return numpy.array(alist)


class FDResult(object):
    """Stores the result of an FD-method computation."""

    def __init__(self, n, mesh, subdiv, fddepth, Lsum, Usum, DUsum, eta,
                  Unorm, L, U, DU):
        self.result = {}
        self.result["n"] = n
        self.result["fddepth"] = fddepth
        self.result["mesh"] = mesh
        self.result["subdiv"] = subdiv
        self.result["Lsum"] = Lsum
        self.result["Usum"] = Usum
        self.result["DUsum"] = DUsum
        self.result["eta"] = eta
        self.result["Unorm"] = Unorm
        self.result["L"] = L
        self.result["U"] = U
        self.result["DU"] = DU
        # Fields in the list go into the CSV file
        self.tablecolumns = ["Lsum", "Usum", "DUsum", "eta", "Unorm", "L",
                             "U", "DU"]

    def filterfields(self, ffilter):
        """Return a list of values of fields named in ffilter."""

        #print(ffilter, [self.result[col] for col in ffilter])
        return [self.result[col] for col in ffilter]

    def _step(self, d):
        """Return result values for FD-method step d for each CSV column."""

        return [self.result[col][d] for col in self.tablecolumns]

    def __len__(self):
        return self.result["fddepth"]

    def __iter__(self):
        for stepnum in range(len(self)):
            yield self._step(stepnum)

    def writecsv(self, filename = ""):
        """Save the result to a CSV file."""

        if filename == "":
            filename = "L%02u.csv" % self.result["n"]
        with open(filename, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csvwriter.writerow(self.tablecolumns)
            for row in self:
                csvwriter.writerow(row)

class LegendreFD(object):
    """Solves Sturm-Liouville equations using the FD-method."""

    def __init__(self, integrator, q):
        self.q = q # Convergence is proved for q(x) that satisfies (1.3) [1].
        self.integrator = integrator # An IntegratorSinc object or compatible.
        self.numclass = self.integrator.numclass
        self.n = None
        self.nu = None

    def numsolve(self, fddepth, n = 0):
        """Solve a Sturm-Liouville problem numerically.

        Solves problem (1.1), (1.2) of [1].

        fddepth is the order of the FD-method (the number of steps).
        n is the number of the desired eigenvalue.

        Returns an FDResult object.
        """
        self.n = n

        # u0 is a normalized solution to the basic equation, see formula
        # (2.13) of [1].
        u0 = lambda x: LegendreP(self.n, x)
        w0 = lambda x: LegendreQ(self.n, x)
        du0 = lambda x: DLegendreP(self.n, x)
        dw0 = lambda x: DLegendreQ(self.n, x)

        K = self.integrator.K # The number of points the mesh has.
        N = self.integrator.N # the number of subdivisions of (a, b).

        # Corrections for the eigenvalue (\lambda_n^{(j)}) for each step of
        # (iteration) the FD-method.
        L = numpy.zeros((fddepth), dtype=self.numclass)
        # Values of corrections for the eigenfunction on the mesh (u_n^{(j)}).
        # U[0] contains values of the Legendre function P_n(x).
        U = numpy.zeros((fddepth, N + 1, 2 * K + 1), dtype=self.numclass)
        # Values of corrections for the derivative of the eigenfunction
        # on the mesh (u'_n^{(j)}). Needed to calculate the residual.
        DU = numpy.zeros((fddepth, N + 1, 2 * K + 1), dtype=self.numclass)
        # Values of the right side of equation (2.5) on the mesh, see
        # formula (2.14).
        F = numpy.zeros((fddepth, N + 1, 2 * K + 1), dtype=self.numclass)
        # Values of the Legendre function Q_n(x) on the mesh.
        W = numpy.zeros((N + 1, 2 * K + 1), dtype=self.numclass)
        # Values of Q'_n(x)on the mesh.
        DW = numpy.zeros((N + 1, 2 * K + 1), dtype=self.numclass)
        # Values of the residual.
        eta = numpy.zeros((fddepth), dtype=self.numclass)
        # Values of the norm of u_n^{(j)}.
        Unorm = numpy.zeros((fddepth), dtype=self.numclass)

        L[0] = n * (n + 1) # See formula (2.13) of [1].

        # Set values for step 0 according to the solution to the basic problem
        # (2.7) [1].
        for i in range(1, N + 1):
            for j in range(2 * K + 1):
                U[0, i, j] = u0(self.integrator.z[i, j])
                W[i, j] = w0(self.integrator.z[i, j])
                DU[0, i, j] = du0(self.integrator.z[i, j])
                DW[i, j] = dw0(self.integrator.z[i, j])

        self.nu = (2 * n + 1) / 2
        # self.nu = 1 / (scipy.integrate.quad(lambda x: u0(x) ** 2,
        #                                     self.integrator.a,
        #                                     self.integrator.b)[0])


        # Approximation for \lambda_n at each step. Lsum[j] equals the sum of
        # L[j] for all preceeding steps.
        Lsum = numpy.array(L, dtype=self.numclass)
        # Mesh values for the approximation for u_n(x) at each step.
        Usum = numpy.array(U, dtype=self.numclass)
        # Mesh values for the approximation for u'_n(x) at each step.
        DUsum = numpy.array(DU, dtype=self.numclass)

        for d in range(1, fddepth):
            logger.info("n = %2u, d = %2u:", n, d)
            # Compute lambda for the step according to (4.4) [1].
            lambda_integral = self.nu * \
                              self.integrator.integr_ab(U[0], U[d - 1], self.q)
            logger.debug(str(lambda_integral))
            L[d] = sum(lambda_integral) # Sum of values for each subinterval.
            logger.debug("L[%u] = %f", d, L[d])

            # Compute F according to (2.14) [1].
            # For every subinterval of (a, b)...
            for i in range(1, N + 1):
                # For each point z_j on the subinterval...
                for j in range(-K, K + 1):
                    F[d, i, K + j] = U[d - 1, i, K + j] *\
                                     self.q(self.integrator.z[i, K + j])
                    for k in range(0, d): # 0..d - 1
                        F[d, i, K + j] -= L[d - k] * U[k, i, K + j]
                    logger.debug("F[%u, %u, %u] = %f" %
                                 (d, i, j, F[d, i, K + j]))

            # Compute corrections for the eigenfunction according to (2.16) [1]
            u_integral = numpy.zeros((N + 1, 2 * K + 1), dtype=self.numclass)
            w_integral = numpy.zeros((N + 1, 2 * K + 1), dtype=self.numclass)
            for j in range(-K, K + 1):
                u_integral[1:, K + j] = self.integrator.integr_az(j, F[d], U[0])
                w_integral[1:, K + j] = self.integrator.integr_az(j, F[d], W)

            # Carry-over values for the subintervals
            for i in range(1, N + 1):
                u_integral[i] += u_integral[i - 1, 2 * K]
                w_integral[i] += w_integral[i - 1, 2 * K]

            # See formula (2.17) [1]. Below W contains the values of Q_n(x)
            # and U[0] contains the values of P_n(x).
            U[d] = W * u_integral - U[0] * w_integral
            DU[d] = DW * u_integral - DU[0] * w_integral

            # Orthogonality. See p. 7 of [1] starting at formula (4.3).
            c_integral = self.nu * self.integrator.integr_ab(U[d], U[0])
            logger.debug("%s %f" % (str(c_integral), sum(c_integral)))

            U[d] -= sum(c_integral) * U[0]
            DU[d] -= sum(c_integral) * DU[0]

            # Sums
            Lsum[d] = Lsum[d - 1] + L[d]
            Usum[d] = Usum[d - 1] + U[d]
            DUsum[d] = DUsum[d - 1] + DU[d]

            # Compute the residual \eta.
            nested_integrand = numpy.zeros((N + 1, 2 * K + 1),
                                           dtype=self.numclass)
            nested_integral = numpy.zeros((N + 1, 2 * K + 1),
                                          dtype=self.numclass)

            for i in range(1, N + 1):
                for j in range(-K, K + 1):
                    nested_integrand[i, K + j] = (Lsum[d] - \
                                                  self.q(self.integrator.z[i,
                                                  K + j])) * U[d, i, K + j]

            for j in range(-K, K + 1):
                nested_integral[1:, K + j] = self.integrator.integr_az(j,
                                            nested_integrand)

            eta_integrand = (1 - self.integrator.z ** 2) * DU[d] + \
                            nested_integral
            eta_integrand *= eta_integrand
            eta[d] = numpy.sqrt(sum(self.integrator.integr_ab(eta_integrand)))
            Unorm[d] = numpy.sqrt(sum(self.integrator.integr_ab(U[d], U[d])))

            # Step log
            logger.info("L^{%2u}_%u = %-2.15f, ||U^{%2u}_%u|| = %-2.15f",
                        d, n, L[d], d, n, Unorm[d])
            logger.info("eta^{%2u}_%u = %2.25f", d, n, eta[d])

        return FDResult(n, K, N, fddepth, Lsum,
                         Usum / numpy.sqrt(2 / (2 * n + 1)),
                         DUsum / numpy.sqrt(2 / (2 * n + 1)), eta, Unorm, L,
                         U, DU)


class FDLogger(object):
    """Outputs the status and results of LegendreFD's work."""

    def __init__(self):
        global logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create console handler.
        ch = logging.FileHandler('legendrefdnum.log')
        # Create formatter.
        formatter = logging.Formatter('%(asctime)s - %(name)13s - \
%(levelname)8s - %(message)s')
        # Add formatter to the console handler.
        ch.setFormatter(formatter)
        # Add the console handler to logger.
        logger.addHandler(ch)

        self.counting = False
        self.tstart = None
        self.tend = None

    def __enter__(self):
        # Start time; needed to log how long our computation takes.
        self.tstart = time.time()
        self.counting = True

        numpy.seterrcall(lambda type_, flag: logger.error("Floating point \
error (%s) with flag %s", type_, flag))
        numpy.seterr(all='call')

        return logger

    def stop_the_clock(self):
        if self.counting:
            self.tend = time.time() # Computation end time.
        self.counting = False

    def __exit__(self, type_, value, traceback):
        self.stop_the_clock()
        logger.info("Total time used: %2.5f s" % (self.tend - self.tstart))
        return False

class FDPlot(object):
    """Plots the results of a LegendreFD computation using matplotlib."""

    def __init__(self, *results):
        self.results = results

    def _makeplot(self, field = "Unorm", func = numpy.log):
        """"Build plot graphic with matplotlib."""
        matplotlib.rc('font', family='serif')
        for res, currmarker in itertools.izip(self.results,
                                              itertools.cycle(['s', '+', 'o',
                                                               'D', '^'])):
            plt.plot([func(x) for x in res.result[field][1:]], color='k',
                     marker=currmarker)

        plt.ylabel(field)
        plt.grid(True)

    def write_eps(self, filename = "", field = "Unorm", func = numpy.log):
        """Show a plot of the results."""
        if filename == "":
            filename = field + ".eps"
        self._makeplot(field, func)
        plt.savefig(filename)
        plt.close()

    def show(self, field = "Unorm", func = numpy.log):
        """Show a plot of the results."""
        self._makeplot(field, func)
        plt.show()
        plt.close()

def frac(a, b, numclass = numpy.longdouble):
    """Return numclass(a) / numclass(b)."""
    return numclass(a) / numclass(b)
