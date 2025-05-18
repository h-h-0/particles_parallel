# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# const.py: define parameters for the simulation

# Simulation parameters
N = 100			# number of particles
box_size = 100.0	# box size in x and y
dt = 0.001		# time step in s
t_max = 10.0		# max. time, simulation ends
v_max = 25.0		# maximum velocity of particles

# Physical paramerers:
m = 1.0			# mass of particle (no unit)
q = 50.0		# charge of particle (no unit)
g = -10.0		# gravitational acceleration (m/s**2)
