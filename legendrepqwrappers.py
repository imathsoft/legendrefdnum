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

"""Wraps lqmn from scipy.special with failover to mpmath.legenq.

As of SciPy version 0.9.0 the function scipy.special.lqmn returns the value of
1.0e300 for argument z such that abs(z) > 1 - 513 * numpy.finfo(type(zed)).eps
(based on tests done on x86_64 Linux systems). This module provides a
wrapper that fails over to the function mpmath.legenq when this problem is
encountered. mpmath.legenq is considerably slower than scipy.special.lqmn
so we can't just use it all the time."""

import numbers

import numpy
import scipy.special

def isnumber(val):
    """Check if val is a number of a built-in or NumPy type."""
    return isinstance(val, numbers.Number) or \
           not numpy.isnan(val)

def valuedebug(f):
    """Print error message if lqmn fails and return nearest "good" value."""
    def newf(n, zed):
        val = f(n, zed)
        if abs(val) >= 1.0e300: # for z: |z| > 1 - 513 * eps lqmn = 1e+300.
            val = f(n, numpy.sign(zed) * \
                    (1 - 513 * numpy.finfo(type(zed)).eps))
        if not isnumber(val):
            print(f.__name__, n, repr(zed), repr(val))

        return val
    return newf

@valuedebug
def LegendreP(n, z):
    """Return Legendre P function of z of order n."""
    # [0 for f, 1 for f'][order][degree]
    return scipy.special.lpmn(0, n, z)[0][0][-1]

try:
    import mpmqath
    mpmath.mp.dps = 50
    print(mpmath.mp)

    # numpy.longdouble == numpy.float128 on 64-bit machines.
    def mpf2np(x, numclass = numpy.longdouble):
        """Convert an mpmath.mpf number to numclass (numpy.longdouble).

        This is a rather fragile hack."""
        return numclass(repr(x)[5:-2])

    def np2mpf(x):
        """Convert a number to mpmath.mpf."""
        return mpmath.mpf(repr(x))

    @valuedebug
    def LegendreQ(n, z):
        """Return Legendre Q function of z of order n."""
        # [0 for f, 1 for f'][order][degree]
        lq = scipy.special.lqmn(0, n, z)[0][0][-1]
        # for z: |z| > 1 - 513 * numpy.finfo(type(zed)).eps the value of
        # lqmn is 1e+300.
        if abs(lq) > 1.0e299 or numpy.isnan(lq):
            lq = mpf2np(mpmath.legenq(n, 0, np2mpf(z)))
        return lq
except ImportError:
    @valuedebug
    def LegendreQ(n, z):
        """Return Legendre Q function of z of order n."""
        # [0 for f, 1 for f'][order][degree]
        return scipy.special.lqmn(0, n, z)[0][0][-1]

def LegendreDeriv(f):
    """Return the derivate of Legendre function f."""
    return lambda n, z: ((n + 1) * f(n + 1, z) - \
                           (n + 1) * z * f(n, z)) / (z ** 2 - 1)

DLegendreP = LegendreDeriv(LegendreP)
DLegendreQ = LegendreDeriv(LegendreQ)
