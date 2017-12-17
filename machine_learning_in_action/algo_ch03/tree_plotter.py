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
def plot_node(node_text, center_point, parent_point, node_type):
    # Draw annotated text information on single declarative node
    create_plot.ax1.annotate(node_text, xy=parent_point, xycoords="axes fraction", xytext=center_point, textcoords="axes fraction", va="center", ha="center", bbox=node_type, arrowprops=arrow_args)

# ============== FUNCTION TO PLOT TEXT BETWEEN PARENT AND CHILD ELEMENTS =============
def plot_mid_text(center_point, parent_point, text_string):
    x_mid = ((parent_point[0] - center_point[0]) / 2.0) + center_point[0]
    y_mid = ((parent_point[1] - center_point[1]) / 2.0) + center_point[1]
    create_plot.ax1.text(x_mid, y_mid, text_string)

# ================= FUNCTION TO PLOT DECISION TREE BASED ON ALL NODES ================
def plot_tree(decision_tree, parent_point, node_text):
    # Get width and height of decision tree
    number_of_leafs = get_number_of_leafs(decision_tree)
    get_tree_depth(decision_tree)

    # Initialize tree string list for nodal iteration
    tree_string = list(decision_tree)[0]
    center_point = (plot_tree.xOff + (1.0 + float(number_of_leafs)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    
    # Plot values of nodal children on tree
    plot_mid_text(center_point, parent_point, node_text)
    plot_node(tree_string, center_point, parent_point, decision_node)
    tree_dictionary = decision_tree[tree_string]
    
    # Decrement y-axis offset on tree
    plot_tree.yOff = plot_tree.yOff - (1.0 / plot_tree.totalD)

    # Nodal iteration for generating branching visualization
    for key in tree_dictionary.keys():
        if type(tree_dictionary[key]).__name__ == "dict":
            plot_tree(tree_dictionary[key], center_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + (1.0 / plot_tree.totalW)
            plot_node(tree_dictionary[key], (plot_tree.xOff, plot_tree.yOff), center_point, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), center_point, str(key))
    
    plot_tree.yOff = plot_tree.yOff + (1.0 / plot_tree.totalD)
    return

# ================ FUNCTION TO CALCULATE NUMBER OF LEAF NODES IN DATA ================
def get_number_of_leafs(decision_tree):
    number_of_leafs = 0
    tree_string = list(decision_tree)[0]
    tree_dictionary = decision_tree[tree_string]

    # Iterate through all decision tree keys and find all leaf nodes (branching ends)
    for key in tree_dictionary.keys():
        if type(tree_dictionary[key]).__name__ == "dict":
            number_of_leafs += get_number_of_leafs(tree_dictionary[key])
        else:
            number_of_leafs += 1
    
    # print("NUMBER OF LEAFS IS: {}\n".format(number_of_leafs))
    return number_of_leafs

# =============== FUNCTION TO CALCULATE DEPTH OF DECISION NODES IN DATA ==============
def get_tree_depth(decision_tree):
    max_depth = 0
    tree_string = list(decision_tree)[0]
    tree_dictionary = decision_tree[tree_string]

    # Iterate through decision tree's keys for tracking decision nodes and depth count
    for key in tree_dictionary.keys():
        if type(tree_dictionary[key]).__name__ == "dict":
            current_depth = get_tree_depth(tree_dictionary[key]) + 1
        else:
            current_depth = 1
        
        # If the current depth is greater than the max, set the max to the current depth
        if current_depth > max_depth:
            max_depth = current_depth

    # print("MAXIMUM DECISION TREE DEPTH IS: {}\n".format(max_depth))
    return max_depth

# ===================== FUNCTION TO RETRIEVE SINGLE DECISION TREE ====================
def retrieve_tree(iterator):
    list_of_trees =    [{"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
                        {"no surfacing": {0: "no", 1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}}]
    
    print("\nDECISION TREE AT INPUT {} IS: {}\n".format(iterator, list_of_trees[iterator]))
    return list_of_trees[iterator]

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

    # create_plot(t0)
    # retrieve_tree(1)

    dt = retrieve_tree(1)

    number_of_leafs = get_number_of_leafs(dt)
    print("NUMBER OF LEAFS IS: {}\n".format(number_of_leafs))

    max_depth = get_tree_depth(dt)
    print("MAXIMUM DECISION TREE DEPTH IS: {}\n".format(max_depth))

    return

if __name__ == "__main__":
    main()