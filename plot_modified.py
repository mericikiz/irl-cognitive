from itertools import product

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from rp1.gridenv import GridEnvironment
#p = P.plot_state_values(ax, self.world, a1, **self.style)



def plot_state_values(ax, env, values, border, **kwargs):
    """
    Plot the given state values of a GridWorld instance.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the state-values should be plotted.
        values: The state-values to be plotted as table
            `[state: Integer] -> value: Float`.
        border: A map containing styling information regarding the state
            borders. All key-value pairs are directly forwarded to
            `pyplot.triplot`.

        All further key-value arguments will be forwarded to
        `pyplot.imshow`.
    """
    #p = ax.imshow(np.reshape(values, (world.size, world.size)), origin='lower', **kwargs)
    p = ax.imshow(np.reshape(values, (env.height, env.width)), origin='lower', **kwargs, interpolation='none')

    if border is not None:
        for i in range(0, env.height + 1):
            ax.plot([-0.5, env.width - 0.5], [i - 0.5, i - 0.5], **border, label=None)

        for i in range(0, env.width + 1):
            ax.plot([i - 0.5, i - 0.5], [-0.5, env.height - 0.5], **border, label=None)

    #        for i in range(0, world.size + 1):
    #        ax.plot([i - 0.5, i - 0.5], [-0.5, world.size - 0.5], **border, label=None)
    #        This line creates vertical lines at positions i - 0.5 on the x-axis. The y-coordinates of the lines span from -0.5 to world.size - 0.5
    #        ax.plot([-0.5, world.size - 0.5], [i - 0.5, i - 0.5], **border, label=None)
    #        This line creates horizontal lines at positions i - 0.5 on the y-axis. The x-coordinates of the lines span from -0.5 to world.size - 0.5
    return p