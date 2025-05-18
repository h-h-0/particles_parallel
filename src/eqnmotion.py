# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# eqnmotion.py: implement the equation of motion of the particles-in-a-box
# as a first order ODE using state vectors so that it can be solved using
# a Runge-Kutta type integrator.

import numpy as np
from const import g, q, m, N

def eqnmotion(s, particles, idx):
    """Computes the derivative of the state vector s for particle idx."""
    r = s[0:2]
    v = s[2:4]
    a = np.array([0.0, g])
    for j in range(N):
        if j == idx:
            continue
        rj = particles[j, 0:2]
        delta_r_vec = r - rj
        delta_r_skl = np.linalg.norm(delta_r_vec)
        a += q**2*delta_r_vec/(m*delta_r_skl**3)
    return np.concatenate([v, a]) 
