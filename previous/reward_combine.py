import numpy as np
from tools import normalize_array
import sys, os # allow us to re-use the framework from the src directory
sys.path.append(os.path.abspath(os.path.join('../irl-maxent-master/src/irl_maxent'))) #from irl-maxent-master.src.irl_maxent
import gridworld as W                       # basic grid-world MDPs
import trajectory as T                      # trajectory generation
import optimizer as O                       # stochastic gradient descent optimizer
import solver as S                          # MDP solver (value-iteration)
import plot as P                            # helper-functions for plotting

class RewardCombiner:

    def __init__(self, name, world, cognitive_control_constant):
        self.name = name
        self.world = world
        self.toCombine = [] #these are individual rewards
        self.combined_value_it = None
        self.combined_value_it_n = None
        self.cc_constant = cognitive_control_constant
        self.naive_combine = None
        self.v1 = None
        self.v2 = None
        self.v1_n = None
        self.v2_n = None
        self.debug = False
        # add weights later maybe : default behaviour, they all weigh the same

    def get_reward(self, time, state):
        return 0

    def naive_reward_compute(self):
        sum = self.toCombine[0].reward_arr
        for i in range(1, len(self.toCombine)):
            sum += self.toCombine[i].reward_arr
        return sum
    def set_toCombine(self, list_of_rewards):
        self.toCombine = list_of_rewards
        self.combine_value_iteration()
        #self.naive_value_iteration()

    def naive_value_iteration(self, eps=1e-5):
        p = self.world.p_transition
        r1 = self.toCombine[0]
        r2 = self.toCombine[1]
        n_states, _, n_actions = p.shape
        delta = np.inf
        v = np.random.rand(n_states)
        while (delta > eps): # update v
            v_old = np.copy(v)
            if self.debug: print(v)
            for i in range(n_states):
                #q s are temporary and only used locally
                q = np.zeros(n_actions) #np.ones(n_actions)/n_actions
                max_value_sys1 = 0
                for j in range(n_actions):
                    possible_next_state = self.world.state_index_transition(i, j)
                    if (r1.indv_value_it[possible_next_state] > max_value_sys1):
                        next_state_sys1 = possible_next_state
                print("next_state_sys1", next_state_sys1)
                for j in range(n_actions):
                    next_state = self.world.state_index_transition(i, j)
                    q[j] = r2.reward_arr[i] - self.cc_constant*(r1.indv_value_it[next_state_sys1]-r1.indv_value_it[next_state]) + r2.time_discount*v[next_state]
                if self.debug: print("state", i, "q array", q)
                chosen_action = np.argmax(q)
                if self.debug: print("chosen_action", chosen_action)
                v[i] = q[chosen_action]
                if self.debug: print("new vi", v[i])
            if self.debug:
                print("last v")
                print(v)
            delta = np.max(np.abs(v_old - v))
        self.naive_combine = v


    def combine_value_iteration(self, eps=1e-4, other_eps=1e-1):
        p = self.world.p_transition
        r1 = self.toCombine[0]
        r2 = self.toCombine[1]
        n_states, _, n_actions = p.shape
        #v1 = np.zeros(n_states) #r1.indv_value_it  #might initialize with individual value iterations
        #v2 = np.zeros(n_states) #r2.indv_value_it #np.zeros(n_states)
        #v1 = r1.reward_arr
        #v2 = r2.reward_arr
        #v1 = r1.indv_value_it
        #v2 = r2.indv_value_it
        v1 = np.random.rand(n_states) # shape as argument
        v2 = np.random.rand(n_states)
        #v2[4] = 10
        delta = np.inf
        num =0
        #other_delta = np.inf
        while (delta > eps) and num <100000 :# or (other_delta > other_eps):  # iterate until convergence
            v_old1 = np.copy(v1) #initialization here
            v_old2 = np.copy(v2)
            #flip = True
            for i in range(n_states):
                #q s are temporary and only used locally
                q1 = np.zeros(n_actions) #np.ones(n_actions)/n_actions
                q2 = np.zeros(n_actions)
                for j in range(n_actions):
                    next_state = self.world.state_index_transition(i, j)
                    #next_state2 = self.world.state_index_transition(i, j)
                    q1[j] = r1.reward_arr[i] + r1.time_discount*v1[next_state]  #reward from current state +
                    q2[j] = r2.reward_arr[i] + r2.time_discount*v2[next_state]
                    #q1 = normalize_array(q1)
                    #q2 = normalize_array(q2)
                #q1 = normalize_array(q1)
                #q2 = normalize_array(q2)
                #q1 = np.divide(q1, np.sum(q1), out=np.zeros_like(q1), where=q1!=0) #q1/np.sum(q1)
                actionstar = np.argmax(self.cc_constant * q1 + q2) #returns index of the max value action
                v1[i] = q1[actionstar]
                v2[i] = q2[actionstar]
                #if flip: v1[i] = q1[actionstar]
                #else: v2[i] = q2[actionstar]
                #flip = not flip
            #v1 = normalize_array(v1)
            #v2 = normalize_array(v2)
            # compute maximum delta
            num = num + 1
            delta = max(np.max(np.abs(v_old1 - v1)), np.max(np.abs(v_old2 - v2)))
            #other_delta = np.max(np.abs(v1 - v2))
        print(num)
        self.combined_value_it = self.cc_constant * v1 + v2 - self.cc_constant * r1.indv_value_it  # cc_constant*(r1.indv_value_it-v1)
        self.v1 = v1#cc_constant * v1 - cc_constant * r1.indv_value_it
        self.v2 = v2#v2 - cc_constant * r1.indv_value_it
        #self.v1_n = normalize_array(v1)
        #self.v2_n = normalize_array(v2)

        #self.combined_value_it_n = normalize_array(self.combined_value_it)
        #self.combined_value_it[self.combined_value_it == 0] = 0.1


