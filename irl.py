''' DESCRIPTION IMPLEMENTATION
Max Entropy IRL Algorithm is described in paper by Ziebert et al. (2008)
and a similar implementation is given in this Jupyter notebook: https://nbviewer.org/github/qzed/irl-maxent/blob/master/notebooks/maxent.ipynb

Algorithm implemented here is modified to account for:
- different environments
- multiple reward functions per environment
- multiple state features
- reward functions containing uncertainty
- an expert agent with cognitive parameters representing risk behaviour and biases in presence of uncertainty
'''

from rp1.evaluation_utils.visualizations_all import Visuals
import rp1.irlutils as irlutils
import rp1.helpers.optimizer as O
import rp1.evaluation_utils.evaluation as E
from rp1.helpers import solver_modified as S

debug = False

class IRL_cognitive():
    def __init__(self, env, cognitive_model, settings, visualize=True):
        self.env = env
        self.cognitive_model = cognitive_model
        self.settings = settings
        self.n_trajectories = settings["Other parameters"][
            "number of expert trajectories"]  # the number of "expert" trajectories
        self.weighting = settings["Other parameters"]["policy weighting"]  # lambda function down-weighting of less optimal actions
        #self.temperature = settings["Other parameters"]["softmax temperature"]
        self.start = settings["Environment"]["start"]
        self.terminal = settings["Environment"]["terminal"]
        self.semi_target = settings["Environment"]["semi target"]
        self.RL_algorithm = settings["Experiment info"]["RL algorithm used"]  # string
        self.eliminate_loops = settings["Other parameters"]["eliminate loops in trajectory"]
        self.visualize=visualize
        self.vis = Visuals(env, cognitive_model, settings, save_bool=visualize, show=False)
        self.mode = settings["Experiment info"]["mode"]

    def generate_trajectories(self, value_final): #can be computed from different given value/q tables
        # given final value is value iteration or q table?
        if self.RL_algorithm == "Value Iteration": just_value = True
        else: just_value = False

        policy_arr = irlutils.stochastic_policy_arr(value_final, self.env, just_value, self.weighting)

        trajectories_with_actions = list(irlutils.generate_trajectories_gridworld(self.n_trajectories, self.env, self.start, self.terminal, self.semi_target, policy_arr))
        if not self.eliminate_loops:
            return irlutils.states(trajectories_with_actions), policy_arr
        else:  # improved trajectories
            trajectories = []
            for tr in trajectories_with_actions:
                trajectories.append(irlutils.erase_loops(list(irlutils.states(tr)), self.semi_target))
            return trajectories, policy_arr

    def perform_irl(self, save_intermediate_guessed_rewards): # for now the expert policy is vanilla value iteration
        if (self.visualize):
            self.vis.visualize_initials() #no calculations actually happen here
            self.vis.info_table(self.settings)

        features = self.env.state_features_one_dim()

        expert_trajectory_states, expert_policy = self.generate_trajectories(self.cognitive_model.value_it_1_and_2_soph_subj_all)

        init = O.Constant(1.0)  # initialize parameters with constant
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))  # optimization strategy
        reward_maxent, p_initial, e_svf, e_features, intermediate_results = irlutils.maxent_irl(self.env, features, self.terminal,
                                                                          expert_trajectory_states, optim, init,
                                                                          self.eliminate_loops, save_intermediate_guessed_rewards)

        if self.visualize:
            joint_time_disc = (self.cognitive_model.time_disc1 + self.cognitive_model.time_disc2) / 2
            self.vis.visualize_initial_maxent(reward_maxent, joint_time_disc, t1="Reward System 1",
                                              t2="Reward System 2", t3="Recovered Reward IRL",
                                              mode=self.mode)  # TODO change
            self.vis.visualize_feature_expectations(e_svf, features, e_features, mode=self.mode)
        if debug: print("done with computing maxent reward")

        value_it_irl = S.value_iteration(self.env.p_transition, reward_maxent, 0.9) #0.75 is a guess that will be constant among trials, this irl method does not estimate the time discount
        agent_trajectory_states, agent_policy = self.generate_trajectories(value_it_irl) #used to be reward_maxent
        optimal_trajectory_states, optimal_st_policy = self.generate_trajectories(self.cognitive_model.simple_v)

        rewards_dict = E.reward_comparison(expert_trajectory_states, agent_trajectory_states, optimal_trajectory_states, self.env.r, self.env.rp_1, self.env.r2)
        cosine_sim_dict = E.policy_comparison(expert_policy, agent_policy, optimal_st_policy)

        optimal_det_policy = S.optimal_policy(self.env, self.cognitive_model.simple_rp_1, (self.cognitive_model.time_disc1 + self.cognitive_model.time_disc2) / 2)

        results_dict = {
            "cosine_sim_dict": cosine_sim_dict,
            "rewards_dict": rewards_dict,
            "expert_st_policy": expert_policy,
            "expert_trajectories": expert_trajectory_states,
            "agent_st_policy": agent_policy,
            "agent_trajectories": agent_trajectory_states,
            "value_it_irl": value_it_irl,
            "optimal_st_policy": optimal_st_policy,
            "optimal_det_policy": optimal_det_policy,
            "optimal_trajectories": optimal_trajectory_states,
            "reward_maxent": reward_maxent,
            "e_svf": e_svf,
            "e_features": e_features,
            "trajectory_feature_expectation": e_features,
            "maxent_feature_expectation": features.T.dot(e_svf),
            "intermediate_results": intermediate_results
        }


        traj_fig_name = lambda name: name+" Trajectories over " + str(self.n_trajectories) + " Samples"
        self.vis.visualize_trajectories(expert_trajectory_states, expert_policy, title=traj_fig_name("Expert"),
                                        save_name="expert_demonstrations",
                                        eliminate_loops=self.eliminate_loops)
        self.vis.visualize_trajectories(agent_trajectory_states, agent_policy, title=traj_fig_name("Agent"),
                                        save_name="agent_trajectories", eliminate_loops=self.eliminate_loops)
        self.vis.visualize_trajectories(optimal_trajectory_states, optimal_st_policy,
                                        title=traj_fig_name("Optimal Stochastic"), save_name="optimal_st_trajectories",
                                        eliminate_loops=self.eliminate_loops)
        self.vis.visualize_optimal_det_policy(optimal_det_policy)

        self.vis.dump_info_to_text(self.cognitive_model.cognitive_calc_dict, results_dict, self.cognitive_model.cognitive_distortion)










