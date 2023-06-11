import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rp1.envVisualizationTools import Env_Visualization
import rp1.irlutils as irlutils
import rp1.helpers.plot_modified as P
import rp1.helpers.solver_modified as S
import mpld3
import numpy as np

import plotly.offline as offline
import plotly.graph_objects as go
import time
from pathlib import Path
import inspect
import copy

plt.rcParams['figure.figsize'] = [9, 5]  # set default figure size
plt.rcParams['image.interpolation'] = 'none'
style = {  # global style for plots
    'border': {'color': 'red', 'linewidth': 0.5},
}

class Visuals():
    def __init__(self, env, cognitive_model, settings, save_bool, show=False):
        self.show = show
        # saving info
        self.save_bool = save_bool
        self.timeStr = time.strftime('%Y%m%d-%H%M%S') #later save inside experiment folders too
        self.save_path = Path("../results", self.timeStr)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        #about experiment
        self.env = env
        self.cognitive_model = cognitive_model
        self.settings = settings

        # figure options
        self.many_colors = ['red', 'green', 'blue', 'orange', 'brown', "goldenrod", "magenta", 'lightpink', 'yellow',
               'darkolivegreen', "darkviolet", "turquoise", "dimgray", "cyan", "cornflowerblue", "limegreen",
               "deeppink", "palevioletred", "lavender", "bisque", "greenyellow", "honeydew", "hotpink", "indianred",
               "indigo", "ivory", "lawngreen", "lightblue", "lightgray", "lightcoral", "lightcyan", "lemonchiffon", "lightgoldenrodyellow"] #just take as many colors as you need
        self.style = {  # global style for plots
            'border': {'color': 'red', 'linewidth': 0.5},
        }

        self.env_visual = Env_Visualization(self.env, self.cognitive_model)

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
        if self.env.tests_dict["test normalization"]: self.visualize_initial_normalized_tests()
        if self.env.tests_dict["test subjective valuation"]: self.visualize_initial_subjective_valuation_tests()
        if self.env.tests_dict["test subjective probability"]: self.visualize_initial_normalized_tests()
        heatmap = self.env_visual.make_pictured_heatmap()
        if self.save_bool: # Save the figure as an HTML file
            offline.plot(heatmap, filename=str(self.save_path / ("env_visual" + '.html')), auto_open=False)
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
        settings = copy.deepcopy(self.settings)

        #fix the lambda  function to adjust for print
        #my_lambda = lambda x: x * 2     print(inspect.getsource(my_lambda))
        my_lambda = settings["Other parameters"]["policy weighting"]
        settings["Other parameters"]["policy weighting"] = inspect.getsource(my_lambda)

        data_list = []

        # Iterate through the inner dictionaries
        for dictionary_name, inner_dict in settings.items():
            # Append the inner dictionary as a list of rows
            rows = [list(row) for row in inner_dict.items()]
            data_list += rows
            #data_list.append([['', ''], ['', '']]) # one empty row after each section for ease of read

        half_length = len(data_list) // 2
        first_half = data_list[:half_length]
        second_half = data_list[half_length:]
        first_half_col1 = [item[0] for item in first_half]
        first_half_col2 = [item[1] for item in first_half]
        second_half_col1 = [item[0] for item in second_half]
        second_half_col2 = [item[1] for item in second_half]

        header = dict(values=['Title', 'Information', 'Title', 'Information'])
        cells = dict(values=[first_half_col1, first_half_col2, second_half_col1, second_half_col2], height=30)

        table = go.Table(header=header, cells=cells)

        layout = go.Layout(
            title="Experiment and Hyperparameter Information",
            height=50*len(data_list),
            margin=dict(l=5, r=2, t=27, b=1)
        )

        fig = go.Figure(data=table, layout=layout)


        # Save the figure as an HTML file
        if self.save_bool:
            offline.plot(fig, filename=str(self.save_path / ("Settings_table" + '.html')), auto_open=False)

        #results_fig.update_layout(margin=dict(l=1, r=1, t=1, b=1))
        #offline.plot(results_fig, filename=save_path + 'results_fig.html', auto_open=False)

    def visualize_trajectories(self, trajectories, policy_array, title, save_name, eliminate_loops):
        #policy_array can be expert policy array or any other
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.title.set_text(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_stochastic_policy(ax, self.env, policy_array, **self.style)
        fig.colorbar(p, cax=cax)

        for t in trajectories:
            P.plot_trajectory(ax, self.env, t, eliminate_loops=eliminate_loops, lw=5, color='white', alpha=0.025)

        fig.tight_layout()
        if self.save_bool: self.save_matplotlib(save_name, fig, html=False)

    def visualize_initial_maxent(self, reward_maxent, mode):
        # mode can be objective, loss sensitive, risk sensitive     TODO handle mode?? tf is that
        fig = plt.figure()
        fig.suptitle("Reward Inference in Mode " + mode)

        ax = fig.add_subplot(131)
        ax.title.set_text('Original Reward System 1')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, self.env.r1, **style)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, self.env.r1, self.cognitive_model.time_disc1), color='red')
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(132)
        ax.title.set_text('Original Reward System 2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, self.env.r2, **style)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, self.env.r2, self.cognitive_model.time_disc2, color='red'))
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text('Recovered Reward')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, reward_maxent, **style)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, reward_maxent, 0.98), color='red') #TODO make it a param?, essentially I want to ignore but it is needed for convergence
        fig.colorbar(p, cax=cax)
        fig.tight_layout()

        if self.save_bool: self.save_matplotlib("Reward Inference mode " + mode, fig, html=False)
        #plt.show()


    def visualize_feature_expectation(self, e_svf, features, e_features, reward_name):
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.title.set_text('Trajectory Feature Expectation')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, e_features, **style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(122)
        ax.title.set_text('MaxEnt Feature Expectation')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, features.T.dot(e_svf), **style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()

        if self.save_bool: self.save_matplotlib("feature expectation " + reward_name, fig, html=False)
        #plt.show()

