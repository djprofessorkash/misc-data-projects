"""
NAME:               tree_plotter.py (data_projects/machine_learning_in_action/algo_ch03/)

DESCRIPTION:        Python plot visualization of decision tree ML algorithms. 

                    All source code is available at www.manning.com/MachineLearningInAction. 

NOTE:               Original source code is Python 2, but my code is Python 3.

CREDIT:             Machine Learning In Action (Peter Harrington)
"""


# ====================================================================================
# ================================ IMPORT STATEMENTS =================================
# ====================================================================================


import matplotlib.pyplot as plt             # Module for MATLAB-like data visualization capability
from time import time as t                  # Package for tracking modular and program runtime


# ====================================================================================
# ============================== INITIALIZING CONSTANTS ==============================
# ====================================================================================


decision_node = dict(boxstyle="sawtooth", fc="0.8")         # Initialize decision nodes (branching nodes)
leaf_node = dict(boxstyle="round4", fc="0.8")               # Initialize leaf nodes (ending nodes)
arrow_args = dict(arrowstyle="<-")                          # Initialize arrow arguments with text


# ====================================================================================
# ==================== HELPER FUNCTIONS FOR PLOTTING VISUALIZATION ===================
# ====================================================================================


# =========================== FUNCTION TO PLOT SINGLE NODE ===========================
def plot_node(node_txt, center_point, parent_point, node_type):
    # Draw annotated text information on single declarative node
    create_plot.ax1.annotate(node_txt, xy=parent_point, xycoords="axes fraction", xytext=center_point, textcoords="axes fraction", va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

# ========================== FUNCTION TO CREATE VISUAL PLOT ==========================
def create_plot(t0):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    # Draw subplot on plot
    create_plot.ax1 = plt.subplot(111, frameon=False)
    
    # Plot decision and leaf nodes, then show visualization
    plot_node("a decision node", (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node("a leaf node", (0.8, 0.1), (0.3, 0.8), leaf_node)

    # Track ending time of program and determine overall program runtime
    t1 = t()
    delta = (t1 - t0) * 1000
    
    print("Real program runtime is {0:.4g} milliseconds.\n".format(delta))
    plt.show()


# ====================================================================================
# ================================= MAIN RUN FUNCTION ================================
# ====================================================================================


def main():
    # Track starting time of program
    t0 = t()

    create_plot(t0)
    return

if __name__ == "__main__":
    main()