# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from itertools import product
#
#
#
# trajectories_out = [
# [0, 0, 5, 0, 5, 5, 5, 10, 5, 5, 0, 5, 10, 5, 5, 6, 11, 6, 5, 10, 10, 15, 15, 20, 20, 21, 22, 23, 24],
#
# [0, 0, 5, 10, 15, 15, 20, 20, 20, 21, 21, 22, 23, 24],
#
# [0, 0, 0, 5, 0, 0, 0, 0, 5, 10, 11, 10, 15, 20, 21, 21, 22, 23, 24],
#
# [0, 5, 10, 15, 15, 16, 17, 16, 15, 15, 10, 15, 20, 20, 15, 16, 21, 21, 21, 22, 23, 23, 24],
#
# [0, 5, 5, 0, 0, 5, 10, 5, 10, 10, 5, 0, 5, 5, 5, 5, 10, 15, 16, 21, 22, 22, 22, 23, 23, 24],
#
# [0, 5, 0, 0, 0, 0, 5, 10, 10, 15, 20, 20, 21, 22, 23, 24],
#
# [0, 5, 0, 0, 5, 10, 15, 20, 20, 20, 21, 20, 20, 20, 21, 21, 21, 22, 21, 21, 22, 23, 24],
#
# [0, 5, 0, 5, 5, 0, 0, 5, 5, 5, 5, 10, 10, 15, 20, 21, 22, 23, 24],
#
# [0, 0, 0, 5, 5, 6, 5, 0, 5, 5, 5, 10, 11, 10, 11, 12, 13, 14, 19, 24],
#
# [0, 0, 5, 5, 10, 5, 10, 11, 16, 21, 22, 23, 24],
#
# [0, 0, 0, 5, 10, 5, 10, 10, 10, 5, 10, 5, 10, 5, 0, 5, 10, 15, 20, 21, 21, 16, 21, 22, 23, 24],
#
# [0, 0, 0, 0, 5, 5, 10, 5, 0, 5, 0, 5, 10, 15, 20, 20, 20, 21, 16, 21, 21, 22, 23, 24],
#
# [0, 5, 10, 11, 10, 5, 5, 0, 1, 6, 5, 10, 15, 16, 15, 20, 21, 20, 15, 20, 20, 21, 22, 22, 22, 17, 18, 23, 24],
#
# [0, 0, 5, 5, 5, 0, 5, 10, 15, 10, 10, 15, 16, 15, 20, 20, 20, 20, 20, 21, 20, 20, 20, 21, 22, 23, 23, 23, 24],
#
# [0, 0, 5, 0, 0, 0, 5, 10, 15, 20, 20, 21, 20, 20, 20, 15, 20, 15, 20, 21, 22, 23, 24],
#
# [0, 0, 0, 5, 10, 10, 15, 20, 21, 16, 17, 22, 23, 23, 24]]
#
# expert_policy_out = np.array([
#  [2.67030961e-01, 2.14877241e-01, 3.03214557e-01, 2.14877241e-01],
#  [9.99984074e-01, 6.26863983e-06, 1.86681358e-06, 7.79012661e-06],
#  [9.99970870e-01, 2.26924369e-10, 1.76986239e-10, 2.91292769e-05],
#  [9.99954601e-01, 1.32244561e-09, 2.17494968e-10, 4.53978686e-05],
#  [4.99977301e-01, 2.26989344e-05, 2.26989344e-05, 4.99977301e-01],
#  [8.17694730e-02, 3.87456716e-01, 2.56197184e-01, 2.74576627e-01],
#  [2.11057145e-01, 3.07277049e-01, 2.11057145e-01, 2.70608662e-01],
#  [1.41230912e-01, 1.60312075e-06, 3.40609477e-05, 8.58733424e-01],
#  [4.99999995e-01, 8.84956966e-11, 9.95543513e-09, 4.99999995e-01],
#  [4.53978687e-05, 2.17494968e-10, 6.34959241e-11, 9.99954602e-01],
#  [2.45265475e-01, 2.36111773e-01, 1.61541966e-01, 3.57080786e-01],
#  [7.56789785e-01, 1.11600958e-01, 9.59900075e-02, 3.56192498e-02],
#  [9.72612723e-01, 8.64573365e-03, 1.00958099e-02, 8.64573365e-03],
#  [2.25916982e-01, 1.86629034e-04, 5.44848605e-05, 7.73841904e-01],
#  [1.39865231e-06, 1.99108421e-08, 5.06961265e-09, 9.99998576e-01],
#  [2.52648067e-01, 2.00967393e-01, 2.52648067e-01, 2.93736473e-01],
#  [2.92362769e-01, 1.64904194e-01, 2.92362769e-01, 2.50370267e-01],
#  [1.71012735e-01, 7.42989341e-02, 1.68912000e-01, 5.85776330e-01],
#  [1.96049790e-01, 7.99249645e-03, 2.59745435e-02, 7.69983170e-01],
#  [3.60605834e-03, 2.39935607e-04, 1.28079459e-03, 9.94873211e-01],
#  [3.35321201e-01, 2.37772173e-01, 2.37772173e-01, 1.89134452e-01],
#  [3.99989699e-01, 1.75942551e-01, 2.48125198e-01, 1.75942551e-01],
#  [4.73609277e-01, 1.45731934e-01, 2.34926855e-01, 1.45731934e-01],
#  [5.72938545e-01, 1.06012069e-01, 2.13718860e-01, 1.07330527e-01],
#  [1.92733648e-01, 7.18939506e-02, 1.92733648e-01, 5.42638754e-01]])
#
# plt.rcParams['figure.figsize'] = [9, 5]  # set default figure size
# style = {  # global style for plots
#             'border': {'color': 'red', 'linewidth': 0.5},
#         }
#
# width = 5  # TODO try with diff dimension after
# height = 5
#
#
# def state_point_to_index(state):
#     x, y = state
#     return y * width + x
# def state_index_to_point(state):
#     x = state % width
#     y = state // width
#     return x, y
#
# def plot_trajectory(ax, trajectory, **kwargs):
#     # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
#     """
#     Plot a trajectory as line.
#
#     Args:
#         ax: The matplotlib Axes instance used for plotting.
#         world: The GridWorld for which the trajectory should be plotted.
#         trajectory: The `Trajectory` object to be plotted.
#
#         All further key-value arguments will be forwarded to
#         `pyplot.tripcolor`.
#
#     """
#     #states = irlutils.states(trajectory)
#     xy = [state_index_to_point(s) for s in trajectory] #here trajectory is a list of states
#     x, y = zip(*xy)
#
#     return ax.plot(x, y, **kwargs)
#
#
# def plot_stochastic_policy(ax, policy, border=None, **kwargs):
#     # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
#     """
#     Plot a stochastic policy.
#
#     Args:
#         ax: The matplotlib Axes instance used for plotting.
#         world: The GridWorld for which the policy should be plotted.
#         policy: The stochastic policy to be plotted as table
#             `[state: Index, action: Index] -> probability: Float`
#             representing the probability p(action | state) of an action
#             given a state.
#         border: A map containing styling information regarding the
#             state-action borders. All key-value pairs are directly forwarded
#             to `pyplot.triplot`.
#
#         All further key-value arguments will be forwarded to
#         `pyplot.tripcolor`.
#     """
#
#     xy = [(x - 0.5, y - 0.5) for y, x in
#           product(range(height + 1), range(width + 1))]  # Generate coordinates for gridlines
#     xy += [(x, y) for y, x in product(range(height), range(width))]  # Generate coordinates for grid cell centers
#
#     t, v = [], []
#     for sy, sx in product(range(height), range(width)):  # for each cell
#         state = state_point_to_index((sx, sy))
#         #print("sx:", sx, "sy", sy, "index", state)
#
#         bl, br = sy * (width + 1) + sx, sy * (width + 1) + sx + 1
#         tl, tr = (sy + 1) * (width + 1) + sx, (sy + 1) * (width + 1) + sx + 1
#         cc = ((width+1) * (height+1)) + sy * width + sx
#
#         # compute triangles
#         t += [(tr, cc, br)]  # action = (1, 0)
#         t += [(tl, bl, cc)]  # action = (-1, 0)
#         t += [(tl, cc, tr)]  # action = (0, 1)
#         t += [(bl, br, cc)]  # action = (0, -1)
#
#         # stack triangle values
#         v += [policy[state, 0]]  # action = (1, 0)
#         v += [policy[state, 1]]  # action = (-1, 0)
#         v += [policy[state, 2]]  # action = (0, 1)
#         v += [policy[state, 3]]  # action = (0, -1)
#
#     x, y = zip(*xy)
#     x, y = np.array(x), np.array(y)
#     t, v = np.array(t), np.array(v)
#
#     ax.set_aspect('equal')
#     ax.set_xticks(range(width))
#     ax.set_yticks(range(height))
#     ax.set_xlim(-0.5, width - 0.5)
#     ax.set_ylim(-0.5, height - 0.5)
#
#     p = ax.tripcolor(x, y, t, facecolors=v, vmin=0.0, vmax=1.0, **kwargs)
#
#     if border is not None:
#         ax.triplot(x, y, t, **border)
#
#     return p
#
#
# def visualize_trajectories(trajectories, policy_array, title="deneme"):
#     #policy_array can be expert policy array or any other
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.title.set_text(title)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     p = plot_stochastic_policy(ax, policy_array, **style)
#     fig.colorbar(p, cax=cax)
#
#     for t in trajectories:
#         plot_trajectory(ax, t, lw=5, color='white', alpha=0.05)
#
#     fig.tight_layout()
#     plt.show()
#
# def traj_test(t1, t2, policy_array, title):
#     # policy_array can be expert policy array or any other
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.title.set_text(title)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     p = plot_stochastic_policy(ax, policy_array, **style)
#     fig.colorbar(p, cax=cax)
#
#
#     plot_trajectory(ax, t1, lw=5, color='black', alpha=1.0)
#     plot_trajectory(ax, t2, lw=5, color='white', alpha=0.5)
#
#     fig.tight_layout()
#     plt.show()
#
# def eliminate_loops(route):
#     working_array = route
#     uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
#     #print("uniqueValues", uniqueValues)
#     #print("indicesList", indicesList)
#     duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
#     #print("duplicate_elements", duplicate_elements)
#     #duplicates = (np.all(duplicate_elements == working_array))  # indices of the duplicates
#     #print("duplicates", duplicates)
#     while not (len(duplicate_elements) == 0):
#         element = duplicate_elements[0]
#         #print("working_array", working_array)
#         #print("element", element)
#         duplicates = (np.where(element == working_array)) #indices of the duplicates
#         #print("duplicates", duplicates)
#         beginning_to_first_occurrence = working_array[0:np.amin(duplicates)]
#         #print("beginning_to_first_occurrence", beginning_to_first_occurrence)
#         last_occurrence_to_end = working_array[np.amax(duplicates):len(working_array)]
#         working_array = np.concatenate((beginning_to_first_occurrence, last_occurrence_to_end))
#         uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
#         duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
#     return working_array
#
# #
# # for num, t in enumerate(trajectories_out):
# #     print("before")
# #     print(t)
# #     print(len(t))
# #     t2 = eliminate_loops(t)
# #     print("after")
# #     print(t2)
# #     print(len(t2))
# #     traj_test(t, t2, expert_policy_out, "loop elimination traj no "+str(num))
#
# improved_trajectories = []
# for t in trajectories_out:
#     improved_trajectories.append(eliminate_loops(t))
#
# visualize_trajectories(improved_trajectories, expert_policy_out)