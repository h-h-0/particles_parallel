# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# sim.py: Main program for the simulation

import numpy as np
import eqnmotion
import box
from const import N, box_size, dt, t_max, v_max
import random
from rk4 import rk4_step
from output import output

outfile = "sim_001.txt"
simbox = box.box(0.0, box_size, 0.0, box_size)

# Allocate two buffers for the state vectors and a bool for buffer switching
particles_0 = np.zeros((N, 4))
particles_1 = np.zeros((N, 4))
link_order = True

# Create a random particle population in the box
for i in range(N):
    particles_0[i, 0] = box_size*random.random()	#  x
    particles_0[i, 1] = box_size*random.random()	#  y
    particles_0[i, 2] = v_max*random.random()		# vx
    particles_0[i, 3] = v_max*random.random()		# vy

t = 0.0
with open(outfile, "w") as f:
    while t < t_max:
# The following logic switches between state vector buffers without
# copying data (by toggling link_order):
        if link_order:
            particles_s_old = particles_0
            particles_s_new = particles_1
        else:
            particles_s_old = particles_1
            particles_s_new = particles_0
        link_order = not link_order
        assert particles_s_old is not particles_s_new, \
               f"Error: old and new particle buffer point to the same object."
        output(f, t, particles_s_old)
        for idx in range(N):
            s_old = particles_s_old[idx, :]
            s_new = s_old + rk4_step(eqnmotion.eqnmotion, s_old,
                                     particles_s_old, idx, dt)
            (where, lambda_inters) = simbox.left_box(s_old, s_new)
            if where != None:
                s_inters = s_old + rk4_step(eqnmotion.eqnmotion, s_old,
                                            particles_s_old, idx,
                                            lambda_inters*dt)
                s_inters = box.box.reflect(s_inters, where)
                s_new = s_inters + rk4_step(eqnmotion.eqnmotion, s_inters,
                                        particles_s_old, idx,
                                        (1.0 - lambda_inters)*dt)
            particles_s_new[idx, :] = s_new
        t += dt
