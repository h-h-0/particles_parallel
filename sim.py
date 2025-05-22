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
from mpi4py import MPI

comm = MPI.COMM_WORLD
WORK = 0
DONE = 1
rank = comm.Get_rank()
size = comm.Get_size()

def dispatcher(comm, particles_0):
    """
    Dispatcher function that distributes the work to the workers. 
    It receives the initial state vector and sends it to the workers.
    It also collects the results and writes them to a file.
    """

    t = 0.0
    outfile = "sim_001.txt"
    particles_1 = np.zeros_like(particles_0)
    link_order = True
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

            # send initial work to workers
            next_idx = 0    # this is the index of the next particle to be sent
            for worker in range(1, size):
                comm.Send(particles_s_old, dest=worker, tag=WORK)
                comm.send(next_idx, dest=worker, tag=WORK)
                next_idx += 1

            # receive and send again until looped through all particles
            j = 0   # this tracks the data received from workers for the number of particles
            status = MPI.Status()
            while j < N:
                result = comm.recv(status=status)
                worker = status.Get_source()
                idx, s_new = result
                particles_s_new[idx, :] = s_new
                j += 1

                if next_idx < N:
                    comm.Send(particles_s_old, dest=worker, tag=WORK)
                    comm.send(next_idx, dest=worker, tag=WORK)
                    next_idx += 1
                else:   # send None to workers to signal end of work
                    comm.Send(np.empty((N, 4), dtype=np.float64), dest=worker, tag=DONE)
                    comm.send(None, dest=worker, tag=DONE)

            particles_0, particles_1 = particles_s_new, particles_s_old
            t += dt

def run(comm):
    status = MPI.Status()
    simbox = box.box(0.0, box_size, 0.0, box_size)

    while True:
        # Receive full particle old state vector from dispatcher
        particles_s_old = np.empty((N, 4), dtype=np.float64)

        comm.Recv(particles_s_old, source=0,  status=status)
        if status.Get_tag() == DONE:
            break
        idx = comm.recv(source=0, status=status)
        if idx is None:
            break

        s_old = particles_s_old[idx, :]
        s_new = s_old + rk4_step(eqnmotion.eqnmotion, s_old, particles_s_old, idx, dt)

        (where, lambda_inters) = simbox.left_box(s_old, s_new)
        if where != None:
            s_inters = s_old + rk4_step(eqnmotion.eqnmotion, s_old,
                                         particles_s_old, idx,
                                         lambda_inters*dt)
            s_inters = box.box.reflect(s_inters, where)
            s_new = s_inters + rk4_step(eqnmotion.eqnmotion, s_inters,
                                         particles_s_old, idx,
                                         (1.0 - lambda_inters)*dt)
        comm.send((idx, s_new), dest=0)

def main():
    if rank == 0:
        particles_0 = np.zeros((N, 4))
        for i in range(N):
            particles_0[i, 0] = box_size * random.random()
            particles_0[i, 1] = box_size * random.random()
            particles_0[i, 2] = v_max * random.random()
            particles_0[i, 3] = v_max * random.random()

        start_time = MPI.Wtime()
        dispatcher(comm, particles_0)
        end_time = MPI.Wtime()

        print(f"run time with parallelization: {end_time - start_time:.3f} seconds")
    else:
        run(comm)

if __name__ == "__main__":
    main()
    print("Simulation finished.")
