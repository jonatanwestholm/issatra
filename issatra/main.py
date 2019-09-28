import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from issatra.models import color_intervals, minimize_spill
from issatra.utils import flatten, optimize

colors = ["r", "g", "b", "y", "m", "c", "k"]

def get_intervals(N=10, start=0, end=10):
    intervals = np.random.random([N, 2]) * (end - start) + start
    intervals = [list(sorted(ij)) for ij in intervals]
    return intervals


def get_discrete_intervals(N=10, start=0, end=10):
    end = end - 1
    intervals = np.random.random([N, 2]) * (end - start) + start
    intervals = [list(sorted(ij)) for ij in intervals]
    intervals = [(int(i), int(j) + 1) for i, j in intervals]
    return intervals


def main():
    np.random.seed(1)
    intervals = get_discrete_intervals(N=100, end=100)
    #interval2color = color_intervals(intervals, num_colors=None, 
    #                                 method="mip", mutex="cliques")
    interval2color = minimize_spill(intervals, num_registers=32)
    plot_intervals(intervals, interval2color)


def plot_intervals(intervals, interval2color=None):
    fig, ax = plt.subplots()

    margin = 0.05
    for idx, (i, j) in enumerate(intervals):
        if interval2color is not None:
            if interval2color[idx] is not None:
                col = "b"
            else:
                col = "r"
        else:
            col = "k"
        rect = Rectangle((i, idx), (j - i) - margin, 1 - margin, color=col)
        ax.add_patch(rect)

    ax.set_xlim([np.min(intervals), np.max(intervals)])
    ax.set_ylim([0, len(intervals)])
    plt.draw()
    plt.show()


if __name__ == '__main__':
    main()