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

import numpy as np
from rp1.visualizations_all import Visuals
import rp1.irlutils as irlutils
import rp1.helpers.optimizer as O
import rp1.evaluation as E
from rp1.helpers import solver_modified as S

debug = False

class IRL_cognitive():
    def __init__(self, env, cognitive_model, settings):
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
        self.vis = Visuals(env, cognitive_model, settings, save_bool=True, show=False)
        self.mode = settings["Experiment info"]["mode"]


    def generate_trajectories(self, value_final): #can be computed from different given value/q tables
        # given final value is value iteration or q table?
        if self.RL_algorithm == "Value Iteration": just_value = True
        else: just_value = False

        policy_arr = irlutils.stochastic_policy_arr(value_final, self.env, just_value, self.weighting)
        # impossible states are just given 0 for all, from now also handle those
        #print("policy array")
        #print(policy_arr)

        trajectories_with_actions = list(irlutils.generate_trajectories_gridworld(self.n_trajectories, self.env, self.start, self.terminal, self.semi_target, policy_arr))
        if not self.eliminate_loops:
            return irlutils.states(trajectories_with_actions), policy_arr
        else:  # improved trajectories
            trajectories = []
            for tr in trajectories_with_actions:
                trajectories.append(irlutils.erase_loops(list(irlutils.states(tr)), self.semi_target))
            return trajectories, policy_arr  # these trajectories can have multiple states, keep in mind

    def generate_demonstrations(self, final_value, visualize=True):
        name = "Expert Demonstrations over " + str(self.n_trajectories) + " trajectories"
        trajectory_states, expert_policy = self.generate_trajectories(final_value) #expert_policy is policy array #TODO more sophisticated loop elimination

        if debug: print("returned trajectories")
        if visualize:
            self.vis.visualize_trajectories(trajectory_states, expert_policy, title=name, save_name="expert_demonstrations", eliminate_loops=self.eliminate_loops)
            if debug: print("visualized trajectories")
        return trajectory_states, expert_policy

    def perform_irl(self, visualize=True): # for now the expert policy is vanilla value iteration
        if (visualize): self.vis.visualize_initials() #no calculations actually happen here
        expert_trajectory_states, expert_policy = self.generate_demonstrations(self.cognitive_model.value_it_1_and_2_soph_subj_all, visualize)

        print("expert_policy")
        print(expert_policy)

        features = self.env.state_features_one_dim()

        # choose our parameter initialization strategy:
        #   initialize parameters with constant
        init = O.Constant(1.0)

        # choose our optimization strategy:
        #   we select exponentiated stochastic gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

        # actually do some inverse reinforcement learning
        reward_maxent, p_initial, e_svf, e_features = irlutils.maxent_irl(self.env, features, self.terminal,
                                                                          expert_trajectory_states, optim, init,
                                                                          self.eliminate_loops)

        if visualize:
            joint_time_disc = (self.cognitive_model.time_disc1 + self.cognitive_model.time_disc2) / 2
            self.vis.visualize_initial_maxent(reward_maxent, joint_time_disc, t1="Reward System 1",
                                              t2="Reward System 2", t3="Recovered Reward IRL",
                                              mode=self.mode)  # TODO change
            self.vis.visualize_feature_expectations(e_svf, features, e_features, mode=self.mode)
        print("REWARD MAXENT")
        print(reward_maxent)
        print("done with computing maxent reward")

        value_it_irl = S.value_iteration(self.env.p_transition, reward_maxent, 0.75) #0.75 is a guess that will be constant among trials, this irl method does not estimate the time discount
        agent_trajectory_states, agent_policy = self.generate_demonstrations(value_it_irl) #used to be reward_maxent
        print("agent_policy")
        print(agent_policy)
        optimal_trajectory_states, optimal_policy = self.generate_demonstrations(self.cognitive_model.simple_v)
        print("optimal_policy")
        print(optimal_policy)


        rewards_dict = E.reward_comparison(expert_trajectory_states, agent_trajectory_states, optimal_trajectory_states, self.env.r, self.env.rp_1, self.env.r2)
        cosine_sim_dict = E.policy_comparison(expert_policy, agent_policy, optimal_policy)


        if visualize:
            if debug: print("visualizing..")
            self.vis.visualize_trajectories(agent_trajectory_states, agent_policy, title="Agent trajectories", save_name="agent_trajectories", eliminate_loops=self.eliminate_loops)
            #self.vis.visualize_policy_similarity(sim_array) # TODO pairwise visual? do if needed

        results_dict = {
            "rewards_dict": rewards_dict,
            "cosine_sim_dict":cosine_sim_dict
        }
        self.vis.display_settings_with_results(results_dict, cognitive_distortion=self.cognitive_model.cognitive_distortion)




#'Original Reward System 1'
#'Original Reward System 2'
#'Recovered Reward'












