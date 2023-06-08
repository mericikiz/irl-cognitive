import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product               # Cartesian product for iterators

# allow us to re-use the framework from the src directory
import sys, os
sys.path.append(os.path.abspath(os.path.join('irl-maxent-master/src/irl_maxent')))
#from irl-maxent-master.src.irl_maxent
import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P                            # helper-functions for plotting

plt.rcParams['figure.figsize'] = [9, 5]     # set default figure size
style = {                                   # global style for plots
    'border': {'color': 'red', 'linewidth': 0.5},
}


world = W.GridWorld(size=5) #deterministic gridworld

# ___________________________vanilla reward______________________________________

vanilla_reward = np.zeros(world.n_states)
vanilla_reward[4] = 1.0
vanilla_reward[1] = -0.9
vanilla_reward[2] = -0.9
vanilla_reward[3] = -0.9
vanilla_reward[6] = -0.9

# set up terminal states
terminal = [4] #where trajectories end


vanilla_value = S.value_iteration(world.p_transition, vanilla_reward, discount=0.9) #no discount

fig = plt.figure()
ax = fig.add_subplot(131)
ax.title.set_text('Original Reward')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, vanilla_reward, **style)
fig.colorbar(p, cax=cax)

ax = fig.add_subplot(132)
ax.title.set_text('Value matrix')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_state_values(ax, world, vanilla_value, **style)
fig.colorbar(p, cax=cax)


def generate_expert_trajectories(world, reward, terminal):
    n_trajectories = 200  # the number of "expert" trajectories
    discount = 0.9  # discount for constructing an "expert" policy
    weighting = lambda x: x ** 50  # down-weight less optimal actions
    start = [0]  # starting states for the expert

    # compute the value-function
    value = S.value_iteration(world.p_transition, reward, discount)

    # create our stochastic policy using the value function
    policy = S.stochastic_policy_from_value(world, value, w=weighting)

    # a function that executes our stochastic policy by choosing actions according to it
    policy_exec = T.stochastic_policy_adapter(policy)

    # generate trajectories
    tjs = list(T.generate_trajectories(n_trajectories, world, policy_exec, start, terminal))

    return tjs, policy


# generate some "expert" trajectories (and its policy for visualization)
trajectories, expert_policy = generate_expert_trajectories(world, vanilla_reward, terminal)

ax = fig.add_subplot(133)
ax.title.set_text('Expert Policy and Trajectories')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
p = P.plot_stochastic_policy(ax, world, expert_policy, **style)
fig.colorbar(p, cax=cax)

for t in trajectories:
    P.plot_trajectory(ax, world, t, lw=5, color='white', alpha=0.025)

fig.tight_layout()
plt.show()