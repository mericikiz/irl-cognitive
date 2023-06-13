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
import rp1.helpers.maxent

class IRL_cognitive():
    def __init__(self, env, cognitive_model, settings):
        self.env = env
        self.cognitive_model = cognitive_model #TODO
        self.settings = settings
        self.n_trajectories = settings["Other parameters"][
            "number of expert trajectories"]  # the number of "expert" trajectories
        self.weighting = settings["Other parameters"]["policy weighting"]  # lambda function down-weighting of less optimal actions
        #self.temperature = settings["Other parameters"]["softmax temperature"]
        self.start = settings["IRL"]["start"]
        self.terminal = settings["IRL"]["terminal"]
        self.semi_target = settings["IRL"]["semi target"]
        self.RL_algorithm = settings["Experiment info"]["RL algorithm used"]  # string
        self.eliminate_loops = settings["Other parameters"]["eliminate loops in trajectory"]
        self.vis = Visuals(env, cognitive_model, settings, save_bool=True, show=False)#rn visualizations are done during initialization
        self.mode = settings["IRL"]["mode"]


    def generate_expert_trajectories(self, value_final): #can be computed from different given value/q tables
        # given final value is value iteration or q table?
        if self.RL_algorithm == "Value Iteration": just_value = True
        else: just_value = False

        policy_arr = irlutils.stochastic_policy_arr(value_final, self.env, just_value, self.weighting)

        policy_execution = irlutils.stochastic_policy_adapter(policy_arr) #returns lambda function
        trajectories_with_actions = list(irlutils.generate_trajectories_gridworld(self.n_trajectories, self.env, policy_execution, self.start, self.terminal, self.semi_target, self.eliminate_loops))
        if not self.eliminate_loops:
            return trajectories_with_actions, policy_arr
        else:
            trajectories = []
            for tr in trajectories_with_actions:
                trajectories.append(irlutils.erase_loops(list(irlutils.states(tr)), self.semi_target))
            return trajectories, policy_arr #these trajectories can have multiple states, keep in mind

    def expert_demonstrations(self, final_value, visualize=True):
        name = "Expert Demonstrations over " + str(self.n_trajectories) + " trajectories"
        trajectories, expert_policy = self.generate_expert_trajectories(final_value) #expert_policy is policy array #TODO loop
        #improved_trajectories = [] #only a list of states for now TODO
        #for t in trajectories:
        #     improved_trajectories.append(irlutils.eliminate_loops(t))
        if visualize:
            self.vis.visualize_trajectories(trajectories, expert_policy, title=name, save_name="expert_demonstrations", eliminate_loops=self.eliminate_loops)
        return trajectories, expert_policy

    def perform_irl(self, visualize=True): # for now the expert policy is vanilla value iteration
        if (visualize): self.vis.visualize_initials() #no calculations actually happen here
        print("expert is using self.cognitive_model.value_it_1_and_2_soph_o")
        expert_trajectories, expert_policy = self.expert_demonstrations(self.cognitive_model.value_it_1_and_2_soph_o, visualize)

        features = self.env.state_features_one_dim()

        # choose our parameter initialization strategy:
        #   initialize parameters with constant
        init = O.Constant(1.0)

        # choose our optimization strategy:
        #   we select exponentiated stochastic gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))

        # actually do some inverse reinforcement learning
        reward_maxent, p_initial, e_svf, e_features = irlutils.maxent_irl(self.env.p_transition, features, self.terminal, expert_trajectories, optim, init, eliminate_loops=self.eliminate_loops)
        print("done with computing maxent reward")
        # Note: this code will only work with one feature per state
        # p_initial = irlutils.initial_probabilities_from_trajectories(self.env.n_states, expert_trajectories, self.eliminate_loops)
        # e_svf = irlutils.compute_expected_svf(self.env.p_transition, p_initial, self.settings["IRL"]["terminal"], reward_maxent)
        # e_features = irlutils.feature_expectation_from_trajectories(features, expert_trajectories, self.eliminate_loops)
        print("done with IRL calculations")
        #np.set_printoptions(suppress=True)
        if visualize:
            joint_time_disc = (self.cognitive_model.time_disc1+self.cognitive_model.time_disc2)/2
            self.vis.visualize_initial_maxent(reward_maxent, joint_time_disc, mode=self.mode) #TODO change
            self.vis.visualize_feature_expectations(e_svf, features, e_features, reward_name=self.mode)














