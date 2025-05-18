# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# output.py: format and write a one-line entry for the current time step
# into the output file.

from const import N
from energy import energy

def output(f, t, particles):
    """Computes the total energy, formats and writes a one-line entry
       for time t into the open file f."""
    E = energy(particles)
    line = f"{t:8.5f} {E:12.3f}"
    for i in range(N):
        s = particles[i, :]
        line += f" {s[0]:8.4f} {s[1]:8.4f} {s[2]:8.4f} {s[3]:8.4f}"
    line += "\n"
    f.write(line)
