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

"""A more extensive example for legendrefdnum.

Try using different potentials for the S-L equation: q1, q2 and q3, as well as
disabling and enabling split points.

All the important output is done to the file legendrefdnum.log. To follow the
output in real time on a Unix-like system use the command
$ tail -f legendrefdnum.log

Note that with the default settings this example takes a long time to finish.
Reduce fddepth to 26 and K to 100 in the code below if you want to see what
its outputm looks like quicker.
"""

import legendrefdnum
from legendrefdnum import frac
from numpy import sqrt, abs, log

def q1(x):
    return x

def q2(x):
    return log(abs(x - frac(5, 12)) * abs(x + frac(1, 3)))

def q3(x):
    return  frac(1, sqrt(abs(x + frac(1, 3)))) +\
             log(abs(x - frac(1, 3)))

with legendrefdnum.FDLogger() as logger: # Having a logger is mandatory
    interval = (-1, 1)
    K = 250 # Mesh density.
    N = 4 # The number of subintervals.
    n_range = [0, 1, 5] # Which eigenvalues to find.
    fddepth = 61 # The number of steps of the FD-method to be performed.

    logger.info("Starting computation for K = %u points on N = %u \
subdibvisions of %s", K, N, repr(interval))

    # Split points at the singularities of q2(x).
    sp_q2 = [-1, -frac(1, 3), 0, frac(5, 12), 1]

    # Initialize the integrator and solver objects.
    intsinc = legendrefdnum.IntegratorSinc(interval[0], interval[1], K, N,
                                           splitpoints=sp_q2)
    legendrefd = legendrefdnum.LegendreFD(intsinc, q2)

    nresults = {}
    for n in n_range:
        #  For each eigenvalue \lambda_n solve then problem numerically.
        nresults[n] = legendrefd.numsolve(fddepth, n)
        logger.info("\nL_%u = %s", n, repr(nresults[n].result["Lsum"][-1]))
        # Save the result to a CSV file.
        nresults[n].writecsv()

# Plot the values of Unorm (the norm of the eigenfunction approximation)
# and eta (the residual) for all eigenvalues and save the plots to EPS files.
plot = legendrefdnum.FDPlot(*nresults.values())
plot.write_eps(field="eta")
plot.write_eps(field="Unorm")
