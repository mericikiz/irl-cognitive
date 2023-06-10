import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rp1.irl import IRL_cognitive
import rp1.plot_modified as P
import mpld3

import plotly as plotly
import plotly.figure_factory as ff
import json
import numpy as np
import plotly.graph_objs as goo
import plotly.subplots as sp
import plotly.offline as offline
import os
import plotly.graph_objects as go
import time
from pathlib import Path

class Visuals():
    def __init__(self, irl_obj, save_bool, show=False):
        self.show = show
        # saving info
        self.save_bool = save_bool
        self.timeStr = time.strftime('%Y%m%d-%H%M%S') #later save inside experiment folders too
        self.save_path = Path("../results", self.timeStr)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        #about experiment
        self.irl_obj = irl_obj #of type IRL_cognitive object
        self.env = irl_obj.env
        self.cognitive_model = irl_obj.cognitive_model

        # figure options
        self.many_colors = ['red', 'green', 'blue', 'orange', 'brown', "goldenrod", "magenta", 'lightpink', 'yellow',
               'darkolivegreen', "darkviolet", "turquoise", "dimgray", "cyan", "cornflowerblue", "limegreen",
               "deeppink", "palevioletred", "lavender", "bisque", "greenyellow", "honeydew", "hotpink", "indianred",
               "indigo", "ivory", "lawngreen", "lightblue", "lightgray", "lightcoral", "lightcyan", "lemonchiffon", "lightgoldenrodyellow"] #just take as many colors as you need

        plt.rcParams['figure.figsize'] = [9, 5]  # set default figure size
        plt.rcParams['image.interpolation'] = 'none'
        self.style = {  # global style for plots
            'border': {'color': 'red', 'linewidth': 0.5},
        }

        if self.env.tests_dict["test_normalization"]: self.visualize_initial_normalized_tests()
        if self.env.tests_dict["test_subjective_valuation"]: self.visualize_initial_subjective_valuation_tests()
        if self.env.tests_dict["test_subjective_probability"]: self.visualize_initial_normalized_tests()



    def save_matplotlib(self, name, fig, html=False):
        if self.save_bool:
            if html:
                html_fig = mpld3.fig_to_html(fig)
                with open(str(self.save_path / (name + '.html')), 'w') as f:
                    f.write(html_fig)
            else:
                fig.savefig(str(self.save_path / (name + '.png')), dpi=200, bbox_inches='tight')
        plt.close(fig) #free up resources

    def visualize_initial_subjective_valuation_tests(self):
        self.visualize_1and2(self.cognitive_model.r1_subj_v, self.cognitive_model.r2_subj_v, name="Subjective Rewards",
                             save_name="subjective_rewards", title="Subjective Rewards")
        self.visualize_1and2(self.cognitive_model.v1_subj_v, self.cognitive_model.v2_subj_v,
                             name="Subjective Rewards Individual Value Iteration",
                             save_name="Subjective_Rewards_Individual_Value_Iteration", title="Subjective Rewards")
        self.visualize_valueiterations_12_and_combined(self.cognitive_model.v1_comb_subj,
                                                       self.cognitive_model.v2_comb_subj,
                                                       self.cognitive_model.value_it_1_and_2_soph_subj,
                                                       save_name="System_1_2_and_Joint_Value_Iterations_After_Combination_Subjective_Rewards",
                                                       title="Subjective Rewards")

    def visualize_initial_subjective_probability_tests(self):
        pass

    def visualize_initial_normalized_tests(self):
        self.visualize_1and2(self.env.r1_n, self.env.r2_n, name="Objective Rewards", save_name="normalized_objective_rewards", title="Normalized")
        self.visualize_1and2(self.cognitive_model.v1_o_n, self.cognitive_model.v2_o_n,
                             name="Objective Individual Value Iteration",
                             save_name="Normalized_Objective_Individual_Value_Iteration", title="Normalized")
        self.visualize_valueiterations_12_and_combined(self.cognitive_model.v1_comb_o_n, self.cognitive_model.v2_comb_o_n,
                                                       self.cognitive_model.value_it_1_and_2_soph_o_n, save_name="System_1_2_and_Joint_Value_Iterations_After_Combination_Normalized", title="Normalized")

    def visualize_initials(self):
        self.info_table()
        self.visualize_1and2(self.env.r1, self.env.r2, name="Objective Rewards", save_name="objective_rewards")
        self.visualize_1and2(self.cognitive_model.v1_o, self.cognitive_model.v2_o, name="Objective Individual Value Iteration", save_name="Objective_Individual_Value_Iteration")
        self.visualize_valueiterations_12_and_combined(self.cognitive_model.v1_comb_o, self.cognitive_model.v2_comb_o, self.cognitive_model.value_it_1_and_2_soph_o, save_name="System_1_2_and_Joint_Value_Iterations_After_Combination")

    def visualize_1and2(self, a1, a2, name, save_name, title="Vanilla"):
        fig = plt.figure()
        fig.suptitle(title)

        ax = fig.add_subplot(121)
        ax.title.set_text(name +' System 1')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a1, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(122)
        ax.title.set_text(name +' System 2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a2, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()

        #html_fig = mpld3.fig_to_html(fig) # Convert the figure to an interactive HTML representation
        self.save_matplotlib(save_name, fig, html=False)
        if self.show: plt.show()

    def visualize_valueiterations_12_and_combined(self, v1, v2, v3, save_name, title="Vanilla"):
        fig = plt.figure()
        fig.suptitle(title)

        ax = fig.add_subplot(131)
        ax.title.set_text('System 1 after Combination')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, v1, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(132)
        ax.title.set_text('System 2 after Combination')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, v2, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text('Joint Value Iteration')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, v3, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()
        if self.show: plt.show()

        #html_fig = mpld3.fig_to_html(fig)
        self.save_matplotlib(save_name, fig, html=False)

    def info_table(self):
        settings = self.irl_obj.settings
        # Create a list to store table data
        table_data = []

        # Iterate through the inner dictionaries
        for dictionary_name, inner_dict in settings.items():
            # Append the inner dictionary as a list of rows
            rows = [list(row) for row in inner_dict.items()]
            table_data.extend(rows)

        table = ff.create_table(table_data, height_constant=10)

        # Update layout properties
        table.update_layout(
            title="Inner Dictionaries",
            height=200 * len(settings),
            # margin=dict(l=1, r=1, t=1, b=1)
        )

        # Save the figure as an HTML file
        if self.save_bool:
            offline.plot(table, filename=str(self.save_path / ("Settings_table" + '.html')), auto_open=False)

        #results_fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
        #offline.plot(results_fig, filename=save_path + 'results_fig.html', auto_open=False)
