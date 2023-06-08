import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import sys, os # allow us to re-use the framework from the src directory
sys.path.append(os.path.abspath(os.path.join('irl-maxent-master/src/irl_maxent'))) #from irl-maxent-master.src.irl_maxent
import plot as P                            # helper-functions for plotting


class Visual():
    def __init__(self, name, world):
        self.name = name
        self.world = world
        plt.rcParams['figure.figsize'] = [9, 5]  # set default figure size
        self.style = {  # global style for plots
            'border': {'color': 'red', 'linewidth': 0.5},
        }

    def visualize_valueiterations_12_and_combined(self, v1, v2, v3):
        fig = plt.figure()

        ax = fig.add_subplot(131)
        ax.title.set_text('System 1 after Combination')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.world, v1, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(132)
        ax.title.set_text('System 2 after Combination')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.world, v2, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text('Joint Value Iteration')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.world, v3, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()
        plt.show()

    def visualize_rewards_1and2(self, r1, r2, name="Reward System"): #not necessarily reward, can also be value
        fig = plt.figure()

        ax = fig.add_subplot(121)
        ax.title.set_text(name +' 1')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.world, r1, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(122)
        ax.title.set_text(name +' 2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.world, r2, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()
        plt.show()

    def visualize_trajectories(self, trajectories, expert_policy):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.title.set_text('Expert Policy and Trajectories')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_stochastic_policy(ax, self.world, expert_policy, **self.style)
        fig.colorbar(p, cax=cax)

        for t in trajectories:
            P.plot_trajectory(ax, self.world, t, lw=5, color='white', alpha=0.025)

        fig.tight_layout()
        plt.show()