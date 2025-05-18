# This file is part of the particle-in-a-box simulation written
# by Lex Wennmacher for educational purpose only. Do not re-distribute.

# box.py: Define class box to deal with confining particles in a box.
# It provides methods to determine, whether a particle is in the box,
# where and when it would leave the box during a simulation time step,
# and to reflect a particle to keep it in the box.

import numpy as np

class box:
    def __init__(self, x_min, x_max, y_min, y_max):
        """Initialises a box object, containing the coordinates of the
           left (x_min), right (x_max), lower (y_min), and upper (x_max)
           wall."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


    def in_box(position):
        """Checks if 'position' is inside the box. Returns True if that is the
           case, and False otherwise"""
        if position[0] < self.x_min:
            return False
        if position[0] > self.x_max:
            return False
        if position[1] < self.y_min:
            return False
        if position[1] > self.y_max:
            return False
        return True


    def left_box(self, s_old, s_new):
        """Given an old (s_old) and a new state vector (s_new), determines
           where and when a particle left the box (self). Returns a tuple
           containing the side where the particle left ("x" or "y") and the
           fraction of the time step between the time of the old and the new
           state vector when this happened. Handles the case that a particle
           evades at a corner, crossing both x and y limits."""
        pos_new = s_new[0:2]
        result = []
        if pos_new[0] < self.x_min:
            pos_old = s_old[0:2]
            pos_diff = pos_new - pos_old
            lambda_inters = (pos_old[0] - self.x_min)/pos_diff[0]
            result.append( ("x", lambda_inters) )
        if pos_new[0] > self.x_max:
            pos_old = s_old[0:2]
            pos_diff = pos_new - pos_old
            lambda_inters = (pos_old[0] - self.x_max)/pos_diff[0]
            result.append( ("x", lambda_inters) )
        if pos_new[1] < self.y_min:
            pos_old = s_old[0:2]
            pos_diff = pos_new - pos_old
            lambda_inters = (pos_old[1] - self.y_min)/pos_diff[1]
            result.append( ("y", lambda_inters) )
        if pos_new[1] > self.y_max:
            pos_old = s_old[0:2]
            pos_diff = pos_new - pos_old
            lambda_inters = (pos_old[1] - self.y_max)/pos_diff[1]
            result.append( ("y", lambda_inters) )
        match len(result):
            case 0:
                return (None, 0.0)
            case 1:
                return result[0]
            case 2:
                if result[0][1] < result[1][1]:
                    return result[0]
                else:
                    return result[1]
            case _:
                raise RuntimeError("left_box() botched")


    def reflect(state, where):
        """Reflects a particle at a box wall (either "x" or "y") by inverting
           the appropriate velocity component of the state vector."""
        if where == "x":
            pattern = np.array([1.0, 1.0, -1.0, 1.0])
        elif where == "y":
            pattern = np.array([1.0, 1.0, 1.0, -1.0])
        else:
            raise RuntimeError("reflect() botched")
        return state*pattern
