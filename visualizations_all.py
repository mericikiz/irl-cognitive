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
import json

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

    def visualize_env(self, value_it):
        heatmap = self.env_visual.make_pictured_heatmap(value_it)
        if self.save_bool:  # Save the figure as an HTML file
            offline.plot(heatmap, filename=str(self.save_path / ("env_visual" + '.html')), auto_open=False)

    def visualize_initials(self):
        #visualizing objective rewards and value iterations is a given
        self.visualize_3_r(self.env.r1, self.env.p1, self.env.r2, self.env.r, save_name="objective_rewards",
                           title="Base Objective Rewards", t1="System 1", t2="System 2", t3="Simple combination of both rewards")
        self.visualize_3_v(self.cognitive_model.v1_o, self.cognitive_model.v2_o, self.cognitive_model.simple_v,
                           save_name="objective_indv_v", title="Objective Individual Value Iterations",
                           t1="Values System 1", t2="Values System 2", t3="Values Joint Simple Reward")
        if self.cognitive_model.subjective:
            self.visualize_env(self.cognitive_model.value_it_1_and_2_soph_subj_all)
            self.visualize_3_r(self.cognitive_model.r1_subj_r, self.cognitive_model.r1_subj_p, self.cognitive_model.r1_subj_all, self.cognitive_model.v1_subj_v,
                                 save_name="subjective_assesment_s1", title="Subjective View of System 1 Rewards, Decision Weights and Utilities",
                                 t1="Subjective Rewards and Decision Weights", t2="Subjective Utilities", t3="Value Iteration on Subjective Assesment")
            self.visualize_3_v(self.cognitive_model.v1_comb_subj_all, self.cognitive_model.v2_comb_subj_all, self.cognitive_model.value_it_1_and_2_soph_subj_all,
                                 save_name="subjective_joint_v",
                                 title="Final Subjective Value Iterations Considering Two Systems Together",
                                 t1="Values System 1", t2="Values System 2", t3="Final Value Iteration")

        else:
            self.visualize_env(self.cognitive_model.value_it_1_and_2_soph_o)
            self.visualize_3_v(self.cognitive_model.v1_comb_o, self.cognitive_model.v2_comb_o,
                               self.cognitive_model.value_it_1_and_2_soph_o,
                               save_name="objective_joint_v",
                               title="Objective Value Iterations Considering Two Systems Together",
                               t1="Values System 1", t2="Values System 2", t3="Final Value Iteration")


    def visualize_3_r(self, a1, a1_p, a2, a3, save_name, title, t1, t2, t3):
        fig = plt.figure()
        fig.suptitle(title)

        ax = fig.add_subplot(131)
        ax.title.set_text(t1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a1, **self.style)

        #probs = np.reshape(a1_p, (self.env.height, self.env.width))
        fig.colorbar(p, cax=cax)
        for i, annotation in enumerate(a1_p):
            if not annotation == 1.0:
                x, y = self.env.state_index_to_point(i)
                annotation = np.round(annotation, 2)
                ax.annotate(str(annotation), xy=(x, y), ha='center', va='center', color='white', fontsize=12)

        ax = fig.add_subplot(132)
        ax.title.set_text(t2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a2, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text(t3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a3, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)  # Reduce the top margin

        #html_fig = mpld3.fig_to_html(fig) # Convert the figure to an interactive HTML representation
        self.save_matplotlib(save_name, fig, html=False)
        if self.show: plt.show()

    def visualize_3_v(self, a1, a2, a3, save_name, title, t1, t2, t3):
        fig = plt.figure()
        fig.suptitle(title)

        ax = fig.add_subplot(131)
        ax.title.set_text(t1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a1, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(132)
        ax.title.set_text(t2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a2, **self.style)
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text(t3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, a3, **self.style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)  # Reduce the top margin

        #html_fig = mpld3.fig_to_html(fig) # Convert the figure to an interactive HTML representation
        self.save_matplotlib(save_name, fig, html=False)
        if self.show: plt.show()

    def display_settings_with_results(self, results_dict):
        # Dump the dictionary to JSON and save it to the specified path
        with open(str(self.save_path / ('results.json')), "w") as json_file:
            json.dump(results_dict, json_file)
        settings = copy.deepcopy(self.settings)

        # fix the lambda  function to adjust for print
        my_lambda = settings["Other parameters"]["policy weighting"]
        settings["Other parameters"]["policy weighting"] = inspect.getsource(my_lambda)
        settings["Results"] = results_dict
        self.info_table(settings)

    def info_table(self, dict_display):
        data_list = []
        # Iterate through the inner dictionaries
        for dictionary_name, inner_dict in dict_display.items():
            # Append the inner dictionary as a list of rows
            rows = [list(row) for row in inner_dict.items()]
            data_list += rows


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

    def visualize_initial_maxent(self, reward_maxent, joint_time_disc, t1, t2, t3, mode):
        # mode can be objective, loss sensitive, risk sensitive     TODO handle mode?? tf is that
        print("visualize_initial_maxent")
        fig = plt.figure()
        fig.suptitle("Reward Inference in Mode " + mode)

        ax = fig.add_subplot(131)
        ax.title.set_text(t1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, self.env.r1, **style)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, self.env.r1, self.cognitive_model.time_disc1), color='red')
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(132)
        ax.title.set_text(t2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, self.env.r2, **style)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, self.env.r2, self.cognitive_model.time_disc2), color='red')
        fig.colorbar(p, cax=cax)

        ax = fig.add_subplot(133)
        ax.title.set_text(t3)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, reward_maxent, **style)
        print("starting performing value iteration on recovered reward with time discount", joint_time_disc)
        P.plot_deterministic_policy(ax, self.env, S.optimal_policy(self.env, reward_maxent, joint_time_disc), color='red') #TODO make it a param?, essentially I want to ignore but it is needed for convergence
        fig.colorbar(p, cax=cax)
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)  # Reduce the top margin

        if self.save_bool: self.save_matplotlib("Reward Inference mode_" + mode, fig, html=False)


    def visualize_feature_expectations(self, e_svf, features, e_features, mode):
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
        fig.subplots_adjust(top=0.85)  # Reduce the top margin
        print("does feature expectation")

        if self.save_bool: self.save_matplotlib("Feature Expectation mode_" + mode, fig, html=False)


    def visualize_policy_similarity(self, similarity):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.title.set_text('Policy Similarity')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        p = P.plot_state_values(ax, self.env, similarity, **style)
        fig.colorbar(p, cax=cax)

        fig.tight_layout()

        if self.save_bool: self.save_matplotlib("Policy_Similarity", fig, html=False)