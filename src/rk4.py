# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# rk4_step.py: perform a single time step for a specified particle using
# standard Runge-Kutta fourth-order coefficients.

def rk4_step(f, s, particles, idx, dt):
    """rk4_step(f, s, particles, idx, dt): perform a Runge-Kutta step
       f: equation of motion
       s: state vector
       particles: NumPy array containing the particles. Opaquely passed to f.
       idx: index of particle to compute rk4_step for
       dt: (time) step size
       rk4_step returns a state vector increment"""

    k1 = dt*f(s, particles, idx)
    k2 = dt*f(s + k1/2.0, particles, idx)
    k3 = dt*f(s + k2/2.0, particles, idx)
    k4 = dt*f(s + k3, particles, idx)
    return (k1 + k4)/6.0 + (k2 + k3)/3.0
