# double_pendulum.py

# Created by Aakash Sudhakar
# (C) 2017


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import odeint

# Lengths of the pendulum rods
L1, L2= 1, 1
# Masses of the pendulum bobs (assuming the pendulum weights are negligible)
M1, M2 = 1, 1
# Gravitational acceleration constant on Earth [m/s^2]
g = 9.81

def derivative(y, t, L1, L2, M1, M2):
    theta1, phi1, theta2, phi2 = y  # Define first and second order derivatives of angle




def make_plot(pos):
    ax.plot([0, x1[pos], x2[pos]], [0, y1[pos], y2[pos]], lw=2, c="k")

    circ_fixed = Circle((0, 0), r/2, fc="k", zorder=10)
    circ_rod1 = Circle((x1[pos], y1[pos]), r, fc="b", ec="b", zorder=10)
    circ_rod2 = Circle((x1[pos], y1[pos]), r, fc="r", ec="r", zorder=10)

    ax.add_patch(circ_fixed)
    ax.add_patch(circ_rod1)
    ax.add_patch(circ_rod2)

    seg_num = 20
    segs = max_trail // seg_num

    for counter in range(seg_num):
        pos_min = pos - (seg_num - counter) * segs

        if pos_min < 0:
            continue

        pos_max = pos_min + segs + 1
        alpha = (counter / seg_num) ** 2
        
        ax.plot(x2[pos_min:pos_max], y2[pos_min:pos_max], c="r", solid_capstyle="butt", lw=2, alpha=alpha)

def main():
    # Creates an image every 'di' amount of time, based on an 'fps' number of iterations
    fps = 10
    di = int(1/fps/dt)
    fig, ax = plt.subplots()

    for i in range(0, t.size, di):
        print(i // di, "/", t.size // di)
        make_plot(i)
    return

if __name__ == "__main__":
    main()