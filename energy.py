# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# energy.py: Compute the total energy of the system. The total energy
# is to be conserved on physical grounds. Variations of the total energy
# with time is due to approximations and numerical effects and is a
# proxy for the quality of the simulation.

import numpy as np
from const import g, m, q, N

def energy(particles):
    """Compute the total energy of the system as
       V + U = sum_i(v_i**2/2 + gh_i + sum_j(k/r_ij)"""

    T = 0.0   # accumulate all energy contributions in T (total energy)
    for i in range(N):
# add kinetic energy of particle
        v_vec = particles[i, 2:4]
        v_sqr = np.dot(v_vec, v_vec)
        T += m*v_sqr/2
# add potential energy of particle
        T += -g*m*particles[i, 1]
# add field potential
        ri = particles[i, 0:2]
        for j in range(N):
            if i == j:
                continue      # no self-energy
            rj = particles[j, 0:2]
            delta_r_vec = rj - ri
            delta_r_skl = np.linalg.norm(delta_r_vec)
            T += 0.5*q**2/delta_r_skl
    return T
