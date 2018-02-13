"""
TITLE: double_pendulum.py
DESCRIPTION: This is a working file of the common chaos model, the double pendulum.

WARNING: In order to generate the animated GIF from collected figures, you must install
the ImageMagick library via Homebrew or other package installers in the command line.
Use "brew install imagemagick" for clean Homebrew installation. 

Created and maintained by Aakash Sudhakar.
(C) October 2017

Improved upon from the Double Pendulum example on SciPython's Blog.
LINK: https://scipython.com/blog/the-double-pendulum/ 
CREDIT: @christian (SciPython Blogger)
"""


# ================================================================================
# ============================== IMPORT STATEMENTS ===============================
# ================================================================================


import os                                   # Library for basic operating system mechanics
import numpy as np                          # Library for simple linear mathematical operations (calculates in C; matrix arithmetic)
import matplotlib.pyplot as plt             # Module for MATLAB-like plotting capability
from matplotlib.patches import Circle       # Module for modelling simple circular dynamics
from scipy.integrate import odeint          # Module for solving systems of differential equations
from glob import glob                       # Library for operating on sets of multiple files
from time import time                       # Module for tracking modular and program runtime


# ================================================================================
# =============================== CLASS DEFINITION ===============================
# ================================================================================


# TODO: Create Wolfram Alpha distribution for correct equation
#       Plug in data points from current model 

# TODO: Seaborn and Dash
class Double_Pendulum(object):

    # ======================== CLASS INITIALIZERS/DECLARATIONS =======================
    # TODO: Allow all important parameters to be user-inputted
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
        self.num_of_segs = 20
        # Addresses for animated GIF model and model figure directory
        self.model_name = "dbl_pdm"
        self.fig_dir = glob("./frames/*.png")

    # ===================== METHOD TO SOLVE DIFFERENTIAL EQUATION ====================
    def calculate_derivative(self, y, t, L1, L2, M1, M2):
        # Define zero- and first-order derivatives of angle (position- and velocity-oriented)
        theta1, theta2, phi1, phi2 = y

        # Create constants to hold repetitive mathematical expressions
        const_cos = np.cos(theta1 - theta2)
        const_sin = np.sin(theta1 - theta2)

        # Calculate the first-order derivatives of angular motion (velocity)
        drv_theta1 = phi1
        drv_theta2 = phi2

        # Create dummy constants to hold hard-to-read summative terms
        # NOTE: These constants were derived by hand and verified online (Link: ???)
        # TODO: Upload differential equation derivation notes to directory and repository
        
        # U1 = self.M2 * self.g * np.sin(theta2) * const_cos
        U1 = -self.g * ((2 * self.M1) * self.M2) * np.sin(theta1)
        # U2 = -self.M2 * const_sin
        U2 = -self.M2 * self.g * np.sin(theta1 - (2 * theta2))
        # U3 = self.L1 * const_cos * (drv_theta1 ** 2)
        U3 = -2 * self.M2 * np.sin(theta1 - theta2)
        # U4 = self.L2 * (drv_theta2 ** 2)
        U4 = self.L2 * (drv_theta2 ** 2)
        # U5 = -((self.M1 + self.M2) * self.g * np.sin(theta1))
        U5 = self.L1 * (drv_theta1 ** 2) * np.cos(theta1 - theta2)
        # U6 = self.L1 * (self.M1 + (self.M2 * (const_sin ** 2)))
        U6 = self.L1 * ((2 * self.M1) + self.M2 - (self.M2 * np.cos((2 * theta1) - (2 * theta2))))

        # V1 = self.M1 + self.M2
        V1 = 2 * np.sin(theta1 - theta2)
        # V2 = self.L1 * const_sin * (drv_theta1 ** 2)
        V2 = self.L1 * (drv_theta1 ** 2) * (self.M1 + self.M2)
        # V3 = -(self.g * np.sin(theta2))
        V3 = self.g * (self.M1 + self.M2) * np.cos(theta1)
        # V4 = self.g * np.sin(theta1) * const_cos
        V4 = self.L2 * self.M2 * (drv_theta2 ** 2) * np.cos(theta1 - theta2)
        # V5 = self.M2 * self.L2 * const_cos * const_sin * (drv_theta2)
        V5 = self.L2 * ((2 * self.M1) + self.M2 - (self.M2 * np.cos((2 * theta1) - (2 * theta2))))
        # V6 = self.L2 * (self.M1 + (self.M2 * (const_sin ** 2)))

        # Calculate the second-order derivatives of angular motion (acceleration)
        # drv_phi1 = (U1 + (U2 * (U3 + U4)) + U5) / U6
        drv_phi1 = (U1 + U2 + (U3 * (U4 + U5))) / U6
        # drv_phi2 = ((V1 * (V2 + V3 + V4)) + V5) / V6
        drv_phi2 = (V1 * (V2 + V3 + V4)) / V5
        
        return drv_theta1, drv_theta2, drv_phi1, drv_phi2

    # ======================= METHOD TO CREATE SIMULATION MODEL ======================
    def make_plot(self, ax, x1, x2, y1, y2, pos, drv_pos, max_trail):
        # Creates plotted axial lines with set weights and colors to set up plot model
        ax.plot([0, x1[pos], x2[pos]], [0, y1[pos], y2[pos]], lw=2, c="k")

        # Creates circles of set dimensions for every pendulum node and adds circles to plot model
        circ_fixed = Circle((0, 0), self.r/2, fc="k", zorder=10)
        circ_rod1 = Circle((x1[pos], y1[pos]), self.r, fc="b", ec="b", zorder=10)
        circ_rod2 = Circle((x2[pos], y2[pos]), self.r, fc="r", ec="r", zorder=10)

        ax.add_patch(circ_fixed)
        ax.add_patch(circ_rod1)
        ax.add_patch(circ_rod2)

        # Creates fading line trail to track pendulum's position in space
        segs = max_trail // self.num_of_segs

        for counter in range(self.num_of_segs):
            pos_min = pos - (self.num_of_segs - counter) * segs

            if pos_min < 0:
                continue

            pos_max = pos_min + segs + 1
            alpha = (counter / self.num_of_segs) ** 2
            
            ax.plot(x2[pos_min:pos_max], y2[pos_min:pos_max], c="r", solid_capstyle="butt", lw=2, alpha=alpha)

        # Centers the modelled image on the fixed circle
        lim_param = self.L1 + self.L2 + self.r
        ax.set_xlim(-lim_param, lim_param)
        ax.set_ylim(-lim_param, lim_param)

        # Sets axes' aspects to be equivalent
        ax.set_aspect("equal", adjustable="box")

        # Dynamically updates axes for plot model
        # TODO: Allow user to specify where frames will be saved
        plt.axis("off")
        plt.savefig("frames/fig_{:04d}.png".format(int(pos / drv_pos)))
        plt.cla()
        return

    # ========================= METHOD TO ANIMATE SIMULATION =========================
    # NOTE: Function will break if ImageMagick is not installed
    def animate_model(self):
        # Create sorted list of simulation figures based on numerical order
        list.sort(self.fig_dir, key = lambda img: int(img.split("_")[1].split(".png")[0]))

        # Iterate through all figures and add addresses to text file
        # TODO: Allow user to specify what filename the frames are stored to
        with open("fig_dir.txt", "w") as file:
            for item in self.fig_dir:
                file.write("{}\n".format(item))

        # Convert text file of figure addresses to animated GIF (ImageMagick)
        # TODO: Allow user to specify which file frames are read from
        os.system("convert @fig_dir.txt {}.gif".format(self.model_name))
        return


# ================================================================================
# =============================== MAIN RUN FUNCTION ==============================
# ================================================================================


def main():
    # Track starting time of running program
    runtime_start = time()

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

    print("\nStarting construction of model.\n\nProcess running...\n")

    # Creates figure of model for each selected position and saves to ./frames/ directory
    # TODO: Current runtime is O(n^2) due to nested for loops; can improve? 
    for pos in range(0, t.size, drv_pos):
        # print(pos // drv_pos, "/", t.size // drv_pos)
        dbl_pdm.make_plot(ax, x1, x2, y1, y2, pos, drv_pos, max_trail)

    # Modifies set of figures into animated GIF on parent directory
    # NOTE: Function call will break if ImageMagick is not installed
    dbl_pdm.animate_model()
    print("Process complete. Model has been constructed and saved to current directory.\n")

    # Track ending time of running program
    runtime_end = time()
    print("Total program runtime is {0:.4g} seconds.\n".format(runtime_end - runtime_start))
    return


if __name__ == "__main__":
    main()