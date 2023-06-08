import numpy as np
from itertools import product               # Cartesian product for iterators
from tools import normalize_array
import sys, os # allow us to re-use the framework from the src directory
sys.path.append(os.path.abspath(os.path.join('irl-maxent-master/src/irl_maxent'))) #from irl-maxent-master.src.irl_maxent
import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P                            # helper-functions for plotting

class Reward:
    def __init__(self, name, time_discount, control_cost=False):
        self.name = name
        self.time_discount = time_discount #time of decision vs when the reward will be obtained matters
        self.control_cost = control_cost #there can also be reward functions for which no control cost exists
        self.cc_function = lambda x:x #cognitive control cost function, rn make it a linear one, in fact it's just the same for now
        self.reward_arr = None
        self.reward_arr_n = None
        self.indv_value_it = None
        self.indv_value_it_n = None
        self.indv_opt_pi = None


    def set_individuals(self, reward_array, world):
        self.reward_arr = reward_array
        #self.reward_arr_n = normalize_array(self.reward_arr)
        self.world = world
        self.indv_value_it = S.value_iteration(self.world.p_transition, self.reward_arr, discount=self.time_discount)
        #self.indv_value_it_n =  normalize_array(self.indv_value_it)
        self.indv_opt_pi = S.optimal_policy_from_value(self.world, self.indv_value_it) #table `[state: Integer] -> action: Integer`.




















    # def cognitive_control_cost(self, optimal_action, forseen_action, current_state): #time of reward ? -> this is more about value
    #     if not self.control_cost:
    #         return 0.0
    #     #a and a* in the case of 1decision, pi and pi* in the case of many decisions
    #     else:
    #         x = self.get_reward(forseen_action, current_state) - self.get_reward(optimal_action, current_state)
    #         return self.cc_function(x)
    #
    #
    # def get_immediate_reward(self, forseen_action, current_state):
    #     pass



