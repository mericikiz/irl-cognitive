from rp1.previous.reward_function import Reward
from rp1.previous.reward_combine import RewardCombiner
import numpy as np
import visualizations

import sys, os # allow us to re-use the framework from the src directory
sys.path.append(os.path.abspath(os.path.join('../irl-maxent-master/src/irl_maxent'))) #from irl-maxent-master.src.irl_maxent
import trajectory as T                      # trajectory generation
import solver as S                          # MDP solver (value-iteration)


class System1and2DeterministicMDP():
    def __init__(self, world, start, terminal, cognitive_control_constant):

        self.world = world
        self.start = start
        self.terminal = terminal

        self.visual = visualizations.Visual("system 1 and 2", world)

        self.punishment = -8
        self.prize = 10
        self.tiny_prize = 6
        self.very_tiny_prize = 0.1

        self.combinedR = RewardCombiner("system 1 and 2", world, cognitive_control_constant)
        self.reward1 = self.make_r1()
        self.reward2 = self.make_r2()
        self.cognitive_control_constant = cognitive_control_constant
        #self.visual.visualize_rewards_1and2(self.reward1.reward_arr_n, self.reward2.reward_arr_n)
        #self.visual.visualize_rewards_1and2(self.reward1.indv_value_it_n, self.reward2.indv_value_it_n, name="Individual Value Iteration ")
        self.combinedR.set_toCombine([self.reward1, self.reward2])

    def make_r1(self):
        reward_array = np.zeros(self.world.n_states)
        reward_array[3] = self.punishment
        reward_array[2] = self.punishment
        reward_array[8] = self.punishment
        reward_array[7] = self.punishment
        reward_array[9] = self.punishment
        #reward_array1[12] = punishment
        #reward_array1[13] = punishment
        #reward_array1[14] = punishment
        reward_array[4] = self.tiny_prize
        reward_array[24] = self.tiny_prize
        reward1 = Reward("system 1", 0.9)
        reward1.set_individuals(reward_array, self.world)
        return reward1

    def make_r2(self):
        reward_array = np.zeros(self.world.n_states)
        #reward_array[24] = 10
        #reward_array[4] = -10
        reward_array[4] = self.prize
        reward_array[3] = self.tiny_prize
        reward_array[2] = self.tiny_prize
        reward_array[8] = self.tiny_prize
        reward_array[7] = self.tiny_prize
        reward_array[9] = self.tiny_prize
        reward_array[24] = self.very_tiny_prize
        # reward_array[3] = very_tiny_prize
        # reward_array[8] = very_tiny_prize
        # reward_array[9] = very_tiny_prize
        # reward_array[2] = tiny_prize
        reward2 = Reward("system 2", 0.7)
        reward2.set_individuals(reward_array, self.world)
        return reward2


    def generate_expert_trajectories(self, value_final):
        n_trajectories = 200  # the number of "expert" trajectories
        weighting = lambda x: x**50  # down-weight less optimal actions

        # create our stochastic policy using the value function
        policy = S.stochastic_policy_from_value(self.world, value_final, w=weighting) #every row  sums to 1 as they should
        # a function that executes our stochastic policy by choosing actions according to it
        policy_exec = T.stochastic_policy_adapter(policy) # -> there is something wrong with this, it just freezes sometimes


        # generate trajectories
        tjs = list(T.generate_trajectories(n_trajectories, self.world, policy_exec, self.start, self.terminal))
        return tjs, policy

    def visualize_all(self):
        self.visual.visualize_rewards_1and2(self.reward1.reward_arr, self.reward2.reward_arr)
        self.visual.visualize_rewards_1and2(self.reward1.indv_value_it, self.reward2.indv_value_it, name="Individual Value Iteration ")

        #self.visual.visualize_rewards_1and2(combinedR.v1, combinedR.v2, name="Value Iteration After Commbining")
        self.visual.visualize_valueiterations_12_and_combined(self.combinedR.v1, self.combinedR.v2, self.combinedR.combined_value_it)
        #self.visual.visualize_valueiterations_12_and_combined(self.reward1.indv_value_it, self.reward2.indv_value_it,self.combinedR.naive_combine)

    def generate_expert_trajectories_sys_1_2(self, visualize=True):
        # generate some "expert" trajectories (and its policy for visualization)
        trajectories, expert_policy = self.generate_expert_trajectories(self.combinedR.combined_value_it)
        if visualize:
            self.visual.visualize_trajectories(trajectories, expert_policy)
        return trajectories, expert_policy




