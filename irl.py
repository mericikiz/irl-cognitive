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

class IRL_cognitive():
    def __init__(self, env, cognitive_model, settings):
        self.env = env
        self.cognitive_model = cognitive_model #TODO
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
        self.vis = Visuals(env, cognitive_model, settings, save_bool=True, show=False)#rn visualizations are done during initialization
        self.mode = settings["Experiment info"]["mode"]


    def generate_trajectories(self, value_final): #can be computed from different given value/q tables
        # given final value is value iteration or q table?
        if self.RL_algorithm == "Value Iteration": just_value = True
        else: just_value = False

        policy_arr = irlutils.stochastic_policy_arr(value_final, self.env, just_value, self.weighting)
        #print("policy array")
        #print(policy_arr)

        #policy_execution = irlutils.stochastic_policy_adapter(policy_arr) #returns lambda function
        trajectories_with_actions = list(irlutils.generate_trajectories_gridworld(self.n_trajectories, self.env, self.start, self.terminal, self.semi_target, policy_arr))
        if not self.eliminate_loops:
            return trajectories_with_actions, policy_arr
        else:
            trajectories = []
            for tr in trajectories_with_actions:
                trajectories.append(irlutils.erase_loops(list(irlutils.states(tr)), self.semi_target))
            return trajectories, policy_arr #these trajectories can have multiple states, keep in mind

    def generate_demonstrations(self, final_value, visualize=True):
        name = "Expert Demonstrations over " + str(self.n_trajectories) + " trajectories"
        trajectories, expert_policy = self.generate_trajectories(final_value) #expert_policy is policy array #TODO loop
        #improved_trajectories = [] #only a list of states for now TODO
        #for t in trajectories:
        #     improved_trajectories.append(irlutils.eliminate_loops(t))
        print("returned trajectories")
        if visualize:
            self.vis.visualize_trajectories(trajectories, expert_policy, title=name, save_name="expert_demonstrations", eliminate_loops=self.eliminate_loops)
            print("visualized trajectories")
        return trajectories, expert_policy

    def perform_irl(self, visualize=True): # for now the expert policy is vanilla value iteration
        if (visualize): self.vis.visualize_initials() #no calculations actually happen here
        print("expert is using self.cognitive_model.value_it_1_and_2_soph_subj_all")
        expert_trajectories, expert_policy = self.generate_demonstrations(self.cognitive_model.value_it_1_and_2_soph_subj_all, visualize)

        features = self.env.state_features_one_dim()

        # choose our parameter initialization strategy:
        #   initialize parameters with constant
        init = O.Constant(1.0)

        # choose our optimization strategy:
        #   we select exponentiated stochastic gradient descent with linear learning-rate decay
        optim = O.ExpSga(lr=O.linear_decay(lr0=0.2))


        # actually do some inverse reinforcement learning
        reward_maxent, p_initial, e_svf, e_features = irlutils.maxent_irl(self.env.p_transition, features, self.terminal,
                                                                          expert_trajectories, optim, init,
                                                                          self.eliminate_loops, self.env.impossible_states)

        if visualize:
            joint_time_disc = (self.cognitive_model.time_disc1 + self.cognitive_model.time_disc2) / 2
            self.vis.visualize_initial_maxent(reward_maxent, joint_time_disc, t1="Reward System 1",
                                              t2="Reward System 2", t3="Recovered Reward IRL",
                                              mode=self.mode)  # TODO change
            self.vis.visualize_feature_expectations(e_svf, features, e_features, mode=self.mode)
        print("REWARD MAXENT")
        print(reward_maxent)
        print("done with computing maxent reward")
        print("agent trajectories")
        agent_trajectories, agent_policy = self.generate_demonstrations(reward_maxent)
        print("optimal trajectories")
        optimal_trajectories, optimal_policy = self.generate_demonstrations(self.cognitive_model.simple_v)

        sim_array, avg_sim = E.policy_comparison(expert_policy, agent_policy)

        rewards_dict = E.reward_comparison(expert_trajectories, agent_trajectories, optimal_trajectories, self.env.r, self.env.rp_1, self.env.r2)
        cosine_sim_dict = E.policy_comparison(expert_policy, agent_policy, optimal_policy)


        if visualize:
            print("visualizing..")


            self.vis.visualize_trajectories(agent_trajectories, agent_policy, title="Agent trajectories", save_name="agent_trajectories", eliminate_loops=self.eliminate_loops)
            self.vis.visualize_policy_similarity(sim_array)

        results_dict = {
            "rewards_dict": rewards_dict,
            "cosine_sim_dict":cosine_sim_dict
        }
        self.vis.display_settings_with_results(results_dict)





#'Original Reward System 1'
#'Original Reward System 2'
#'Recovered Reward'












