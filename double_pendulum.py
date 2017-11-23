"""
double_pendulum.py

Created and maintained by Aakash Sudhakar
(C) 2017

Improved upon from the Double Pendulum example on SciPython's Blog.
Link: https://scipython.com/blog/the-double-pendulum/ 
Credit: @christian (SciPython Blogger)
"""

# ================================================================================
# ============================== IMPORT STATEMENTS ===============================
# ================================================================================

import os                                   # Library for basic operating system mechanics
import glob                                 # Library for aggregating multiple files
import numpy as np                          # Library for simple linear mathematical operations
import matplotlib.pyplot as plt             # Module for MATLAB-like plotting capability
from matplotlib.patches import Circle       # Module for modelling simple circular dynamics
from scipy.integrate import odeint          # Module for solving systems of differential equations


# ================================================================================
# =============================== CLASS DEFINITION ===============================
# ================================================================================

class Double_Pendulum(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    def __init__(self):
        # Lengths of the pendulum rods
        self.L1 = 1
        self.L2 = 1
        # Masses of the pendulum bobs (assuming the pendulum weights are negligible)
        self.M1 = 1
        self.M2 = 1
        # Parameters of time spacing and range for plot model
        self.t_max = 20
        self.dt = 0.01
        # Gravitational acceleration constant on Earth [m/s^2]
        self.g = 9.81
        # Nodal radii and tail trace for modelling pendulum movement across time
        self.r = 0.05
        self.t_trail = 1
        # Relative maximum framerate for modelling speed
        self.fps = 10
        # Relative number of framed segments for pendulum node tracer
        self.seg_num = 20
        # Addresses for animated GIF model and model figure directory
        self.model_name = "dbl_pdm"
        self.fig_dir = glob.glob("./frames/*.png")

    # ===================== METHOD TO SOLVE DIFFERENTIAL EQUATION ====================
    def calculate_derivative(self, y, t, L1, L2, M1, M2):
        # Define zero- and first-order derivatives of angle (position- and velocity-oriented)
        theta1, phi1, theta2, phi2 = y

        const_cos = np.cos(theta1 - theta2)
        const_sin = np.sin(theta1 - theta2)

        # Calculate the first- and second-order derivatives of angle (velocity- and acceleration-oriented)
        drv_theta1 = phi1
        drv_theta2 = phi2
        drv_phi1 = ((self.M2 * self.g * np.sin(theta2) * const_cos) - ((self.M2 * const_sin) * ((self.L1 * (drv_theta1 ** 2) * const_cos) + (self.L2 * (drv_theta2 ** 2)))) - ((self.M1 + self.M2) * self.g * np.sin(theta1))) / (self.L1 * (self.M1 + (self.M2 * (const_sin ** 2))))
        drv_phi2 = (((self.M1 + self.M2) * ((self.L1 * (drv_theta1 ** 2) * const_sin) - (self.g * np.sin(theta2)) - (self.g * np.sin(theta1) * const_cos))) + (self.M2 * self.L2 * (drv_theta2 ** 2) * const_cos * const_sin)) / (self.L2 * (self.M1 + (self.M2 * (const_sin ** 2))))

        return drv_theta1, drv_theta2, drv_phi1, drv_phi2

    # ======================= METHOD TO CREATE SIMULATION MODEL ======================
    def make_plot(self, ax, x1, x2, y1, y2, pos, drv_pos, max_trail):
        # Creates plotted axial lines with set weights and colors to set up plot model
        ax.plot([0, x1[pos], x2[pos]], [0, y1[pos], y2[pos]], lw=2, c="k")

        # Creates circles of set dimensions for every pendulum node and adds circles to plot model
        circ_fixed = Circle((0, 0), self.r/2, fc="k", zorder=10)
        circ_rod1 = Circle((x1[pos], y1[pos]), self.r, fc="b", ec="b", zorder=10)
        circ_rod2 = Circle((x1[pos], y1[pos]), self.r, fc="r", ec="r", zorder=10)

        ax.add_patch(circ_fixed)
        ax.add_patch(circ_rod1)
        ax.add_patch(circ_rod2)

        # Creates fading line trail to track pendulum's position in space
        segs = max_trail // self.seg_num

        for counter in range(self.seg_num):
            pos_min = pos - (self.seg_num - counter) * segs

            if pos_min < 0:
                continue

            pos_max = pos_min + segs + 1
            alpha = (counter / self.seg_num) ** 2
            
            ax.plot(x2[pos_min:pos_max], y2[pos_min:pos_max], c="r", solid_capstyle="butt", lw=2, alpha=alpha)

        # Centers the modelled image on the fixed circle
        ax.set_xlim(-(self.L1 + self.L2 + self.r), self.L1 + self.L2 + self.r)
        ax.set_ylim(-(self.L1 + self.L2 + self.r), self.L1 + self.L2 + self.r)

        # Sets axes' aspects to be equivalent
        ax.set_aspect("equal", adjustable="box")

        # Dynamically updates axes for plot model
        plt.axis("off")
        plt.savefig("frames/fig_{:04d}.png".format(int(pos / drv_pos)))
        plt.cla()

        return

    # ========================= METHOD TO ANIMATE SIMULATION =========================
    def animate_model(self):
        list.sort(self.fig_dir, key = lambda img: int(img.split("_")[1].split(".png")[0]))

        with open("fig_dir.txt", "w") as file:
            for item in self.fig_dir:
                file.write("%s\n" % item)

        os.system("convert @fig_dir.txt {}.gif".format(self.model_name))

        return



# ================================================================================
# =============================== MAIN RUN FUNCTION ==============================
# ================================================================================

def main():
    # Initialize instance of double pendulum class to run modeling operations
    dbl_pdm = Double_Pendulum()

    # Create time steps and parameters by which model is created
    t = np.arange(0, dbl_pdm.t_max + dbl_pdm.dt, dbl_pdm.dt)

    # Initial conditions for solving the differential equations of motion
    y0 = [np.pi/2, 0, np.pi/2, 0]

    # Numerical integration to solve the differential equations of motion
    y = odeint(dbl_pdm.calculate_derivative, y0, t, args=(dbl_pdm.L1, dbl_pdm.L2, dbl_pdm.M1, dbl_pdm.M2))

    # Define angular constants as functions of time
    theta1 = y[:, 0]
    theta2 = y[:, 2]

    # Create relative Cartesian positions of pendulum bobs
    x1 = dbl_pdm.L1 * np.sin(theta1)
    x2 = x1 + (dbl_pdm.L2 * np.sin(theta2))
    y1 = -(dbl_pdm.L1 * np.cos(theta1))
    y2 = y1 - (dbl_pdm.L2 * np.cos(theta2))

    # Declare max tracer trail distance
    max_trail = int(dbl_pdm.t_trail / dbl_pdm.dt)

    # Parameters for creating figure snapshots for animated GIF
    drv_pos = int((dbl_pdm.fps * dbl_pdm.dt) ** -1)
    fig, ax = plt.subplots()

    print("Starting construction of model.\n\nProcess running...\n")

    # Creates figure of model for each selected position and saves to ./frames/ directory
    for pos in range(0, t.size, drv_pos):
        # print(pos // drv_pos, "/", t.size // drv_pos)
        dbl_pdm.make_plot(ax, x1, x2, y1, y2, pos, drv_pos, max_trail)

    # Modifies set of figures into animated GIF on parent directory
    dbl_pdm.animate_model()
    print("Process complete. Model has been constructed and saved to current directory.")

    return

if __name__ == "__main__":
    main()
