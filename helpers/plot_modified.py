import copy
from itertools import product
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt



normalize=True

def weighted_normalize(data):
    data = data**2

    negative_range = (-1, 0)
    positive_range = (0, 1)


    # max_positive = 0.0001
    # max_negative = 0.0001
    # if np.any(data > 0):
    #     max_positive = np.amax(data[data > 0])
    # if np.any(data < 0):
    #     max_negative = np.abs(np.amin(data[data < 0]))
    if np.any(data >= 0):
        max_positive = np.amax(data[data > 0])
        data[data > 0] /= max_positive
    if np.any(data < 0):
        max_negative = np.amin(data[data < 0])
        data[data < 0] /= (-1 * max_negative)

    # Apply the normalization range
    norm = Normalize(vmin=negative_range[0], vmax=positive_range[1])
    return data, norm

def two_sided_normalize(data):
    negative_range = (-1, 0)
    positive_range = (0, 1)

    # Normalize the data
    norm = Normalize(vmin=negative_range[0], vmax=positive_range[1])
    # Find the maximum positive value and maximum negative value
    if np.any(data >= 0):
        max_positive = np.amax(data[data > 0])
        data[data > 0] /= max_positive
    if np.any(data < 0):
        max_negative = np.amin(data[data < 0])
        data[data < 0] /= (-1 * max_negative)
    return data, norm


def plot_state_values(ax, env, values, border, **kwargs): #static heatmap only showing values
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
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
    data = copy.deepcopy(values)
    if normalize:
        data, norm = two_sided_normalize(data)

        data = np.reshape(data, (env.height, env.width))
        #print(data)

        # Plot the heatmap
        p = ax.imshow(data, cmap='Spectral', norm=norm, origin='lower', **kwargs, interpolation='none')
    else:
        data = np.reshape(values, (env.height, env.width))
        p = ax.imshow(data, origin='lower', **kwargs, interpolation='none')

    if border is not None:
        for i in range(0, env.height + 1):
            ax.plot([-0.5, env.width - 0.5], [i - 0.5, i - 0.5], **border, label=None)
        #        This line creates vertical lines at positions i - 0.5 on the x-axis. The y-coordinates of the lines span from -0.5 to world.size - 0.5

        for i in range(0, env.width + 1):
            ax.plot([i - 0.5, i - 0.5], [-0.5, env.height - 0.5], **border, label=None)
        #        This line creates horizontal lines at positions i - 0.5 on the y-axis. The x-coordinates of the lines span from -0.5 to world.size - 0.5

    return p

def plot_stochastic_policy(ax, env, policy, border=None, **kwargs):
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    """
    Plot a stochastic policy.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the policy should be plotted.
        policy: The stochastic policy to be plotted as table
            `[state: Index, action: Index] -> probability: Float`
            representing the probability p(action | state) of an action
            given a state.
        border: A map containing styling information regarding the
            state-action borders. All key-value pairs are directly forwarded
            to `pyplot.triplot`.

        All further key-value arguments will be forwarded to
        `pyplot.tripcolor`.
    """
    width = env.width
    height = env.height

    xy = [(x - 0.5, y - 0.5) for y, x in product(range(height + 1), range(width + 1))] # Generate coordinates for gridlines
    xy += [(x, y) for y, x in product(range(height), range(width))]  # Generate coordinates for grid cell centers

    t, v = [], []
    for sy, sx in product(range(env.height), range(env.width)): #for each cell
        state = env.state_point_to_index((sx, sy))

        bl, br = sy * (width + 1) + sx, sy * (width + 1) + sx + 1
        tl, tr = (sy + 1) * (width + 1) + sx, (sy + 1) * (width + 1) + sx + 1
        cc = ((width + 1) * (height + 1)) + sy * width + sx

        # compute triangles
        t += [(tr, cc, br)]                 # action = (1, 0)
        t += [(tl, bl, cc)]                 # action = (-1, 0)
        t += [(tl, cc, tr)]                 # action = (0, 1)
        t += [(bl, br, cc)]                 # action = (0, -1)

        # stack triangle values
        v += [policy[state, 0]]             # action = (1, 0)
        v += [policy[state, 1]]             # action = (-1, 0)
        v += [policy[state, 2]]             # action = (0, 1)
        v += [policy[state, 3]]             # action = (0, -1)

    x, y = zip(*xy)
    x, y = np.array(x), np.array(y)
    t, v = np.array(t), np.array(v)

    ax.set_aspect('equal')
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)

    # Generate a colormap using the name 'Spectral'
    cmap = plt.cm.get_cmap('Spectral')


    p = ax.tripcolor(x, y, t, facecolors=v, cmap=cmap, vmin=-1.0, vmax=1.0, **kwargs)

    if border is not None:
        ax.triplot(x, y, t, **border)

    return p

def get_custom_policy_colors(v):
    cmap = plt.cm.get_cmap('Spectral')
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Get the color for a specific value
    color = sm.to_rgba(v)
    #print("color for ", v, " is ", color)
    return [[color]]



def plot_trajectory(ax, env, trajectory_states, eliminate_loops, **kwargs):
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    """
    Plot a trajectory as line.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the trajectory should be plotted.
        trajectory: The `Trajectory` object to be plotted.

        All further key-value arguments will be forwarded to
        `pyplot.tripcolor`.

    """
    xy = [env.state_index_to_point(s) for s in trajectory_states]
    x, y = zip(*xy)

    return ax.plot(x, y, **kwargs)


def plot_deterministic_policy(ax, env, policy, **kwargs):
    """
    Plot a deterministic policy as arrows.

    Args:
        ax: The matplotlib Axes instance used for plotting.
        world: The GridWorld for which the policy should be plotted.
        policy: The policy to be plotted as table
            `[state: Index] -> action: Index`.

        All further key-value arguments will be forwarded to
        `pyplot.arrow`.
    """
    arrow_direction = [(0.33, 0), (-0.33, 0), (0, 0.33), (0, -0.33)]

    for state in range(env.n_states):
        cx, cy = env.state_index_to_point(state)
        dx, dy = arrow_direction[policy[state]]
        ax.arrow(cx - 0.5 * dx, cy - 0.5 * dy, dx, dy, head_width=0.1, **kwargs)