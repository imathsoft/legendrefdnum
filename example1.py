#!/usr/bin/env python
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

"""A simple use example for legendrefdnum."""

from __future__ import print_function

import legendrefdnum

def q(x):
    return x

with legendrefdnum.FDLogger() as logger: # Having a logger is mandatory
    a = -1
    b = 1
    K = 100  # Mesh density.
    n = 1 # Which eigenvalue to find.
    fddepth = 20 # The number of steps of the FD-method to be performed.

    print("Started.")

    # Initialize the integrator and solver objects.
    intsinc = legendrefdnum.IntegratorSinc(a, b, K)
    legendrefd = legendrefdnum.LegendreFD(intsinc, q)

    # Solve the problem numerically.
    resultobj = legendrefd.numsolve(fddepth, n)

    print("lambda_{%u} = %.8f" % (n, resultobj.result["Lsum"][fddepth - 1]))

# Plot Unorm. Note how it oscillates for q(x) = x. That is because for every
# other j L[j] is zero.
plot = legendrefdnum.FDPlot(resultobj)
plot.show()
