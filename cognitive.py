import numpy as np
from rp1.helpers import solver_modified as S

class Cognitive_model():

    def __init__(self, env, alpha, beta, kappa, eta, gamma1, gamma2, cc_constant, baseline, subjective=False):

        # COGNITIVE PARAMETERS
        self.cc_constant = cc_constant # linear term
        self.cc_cost = lambda x: x*self.cc_constant #cc_constant is cognitive_control_cost function, linear for now
        self.time_disc1 = gamma1
        self.time_disc2 = gamma2
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.eta = eta
        self.baseline = baseline
        # baseline reward and baseline probability depend on individual calculation

        # ENVIRONMENT, REWARDS AND VALUES
        self.env = env
        self.deterministic =  self.env.deterministic
        self.subjective = subjective
        self.impossible_states = self.env.impossible_states
        self.possible_actions_from_state = self.env.possible_actions_from_state

        self.reward_arr1_o = np.copy(self.env.r1) #o stands for objective
        self.reward_arr2_o = np.copy(self.env.r2)
        self.simple_p1 = np.copy(self.env.p1)
        self.simple_rp_1 = np.copy(self.env.rp_1)
        self.simple_r = np.copy(self.env.r)

        # from the agent's perspective, how much is 1 unit of reward is worth on average depending on cognitive variable values
        self.cognitive_distortion = 1.0

        self.v2_o = S.value_iteration(self.env.p_transition, self.reward_arr2_o, discount=self.time_disc2, possible_actions_from_state=self.possible_actions_from_state) #v2 is always deterministic here, assumption caused by experiment design

        if self.deterministic:
            self.v1_o = S.value_iteration(self.env.p_transition, self.reward_arr1_o, discount=self.time_disc1, possible_actions_from_state=self.possible_actions_from_state)
            self.simple_v = S.value_iteration(self.env.p_transition, self.simple_r, discount=(self.time_disc1+self.time_disc2)/2, possible_actions_from_state=self.possible_actions_from_state)
        else:
            self.v1_o = S.uncertainty_value_iteration(self.env.p_transition, self.reward_arr1_o, reward_prob=self.simple_p1, discount=self.time_disc1, possible_actions_from_state=self.possible_actions_from_state)
            self.simple_v = S.uncertainty_value_iteration(self.env.p_transition, self.simple_r, reward_prob=self.simple_p1, discount=(self.time_disc1 + self.time_disc2)/ 2, possible_actions_from_state=self.possible_actions_from_state)

        #_____SUBJECTIVE STUFF_____
        if self.subjective: #assumed it is not deterministic when it's subjective
            self.cognitive_distortion = self.compute_cognitive_distortion()
            vectorized_func = np.vectorize(self.subjective_reward)
            self.r1_subj_r = vectorized_func(self.reward_arr1_o)
            vectorized_func = np.vectorize(self.subjective_probability)
            self.r1_subj_p = vectorized_func(self.simple_p1)
            self.r1_subj_all = np.multiply(self.r1_subj_p, self.r1_subj_r) #not used rn
            self.v1_subj_v = S.uncertainty_value_iteration(self.env.p_transition, self.r1_subj_r, self.r1_subj_p, discount=self.time_disc1, possible_actions_from_state=self.possible_actions_from_state)
            self.v1_comb_subj_all = self.v2_comb_subj_all = None
            self.value_it_1_and_2_soph_subj_all = None
            self.all_subjective()
        else:  # only system 1, system 2 at play
            self.v1_comb_o = None
            self.v2_comb_o = None
            self.value_it_1_and_2_soph_o = None  # do I need this? I dont think so
            self.combine_value_iteration()  # checks for deterministic

        print("debug")


    def compute_cognitive_distortion(self):
        r = self.subjective_reward(self.cognitive_distortion)
        list = np.arange(0.0, 1.0, 0.02)
        prob_avg = 0
        for i in list:
            self.subjective_probability(i)
            prob_avg += i
        prob_avg = prob_avg/len(list)
        self.cognitive_distortion = r*prob_avg


    def combine_value_iteration_uncertainty(self, r1_ref, r2_ref, p1, eps=1e-5):
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        flip = True # interleave the update of two value iterations
        v1 = np.random.rand(n_states) # shape as argument
        v1[self.impossible_states] = 0
        v2 = np.random.rand(n_states) # random initialization here
        v2[self.impossible_states] = 0
        delta = np.inf
        num = 0
        print("impossible_states", self.impossible_states)
        while (delta > eps) and num <10000:
            v_old1 = np.copy(v1)
            v_old2 = np.copy(v2)
            for i in range(n_states):
                q1 = np.zeros(n_actions)
                q2 = np.zeros(n_actions)
                for j in self.possible_actions_from_state[i]:  # don't even do the computation for actions with no effect
                    next_state = self.env.state_index_transition(i, j)
                    q1[j] = r1_ref[i]*p1[i] + self.time_disc1*v1[next_state]
                    q2[j] = r2_ref[i] + self.time_disc2*v2[next_state]
                actionstar = np.argmax(self.cc_constant * q1 + q2) #returns index of the max value action
                if flip: v1[i] = q1[actionstar]
                else: v2[i] = q2[actionstar]
                flip = not flip
            num = num + 1 # to prevent taking too long
            delta = max(np.max(np.abs(v_old1 - v1)), np.max(np.abs(v_old2 - v2)))
        #print("uncertainty value iteration took", num, "steps to converge")
        return v1, v2

    def all_subjective(self):
        v1, v2 = self.combine_value_iteration_uncertainty(self.r1_subj_r, self.reward_arr2_o, self.r1_subj_p)
        self.value_it_1_and_2_soph_subj_all = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_subj_v
        self.v1_comb_subj_all = v1
        self.v2_comb_subj_all = v2

    def combine_value_iteration(self):
        if self.env.deterministic:
            v1, v2 = self.combine_value_iteration_deterministic(self.reward_arr1_o, self.reward_arr2_o)
            self.value_it_1_and_2_soph_o = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_o
            self.v1_comb_o = v1
            self.v2_comb_o = v2
        else:
            v1, v2 = self.combine_value_iteration_uncertainty(self.reward_arr1_o, self.reward_arr2_o, self.simple_p1)
            self.value_it_1_and_2_soph_o = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_o
            self.v1_comb_o = v1
            self.v2_comb_o = v2

    def combine_value_iteration_deterministic(self, r1_ref, r2_ref, eps=1e-5):
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        flip = True # interleave the update of two value iterations
        v1 = np.random.rand(n_states) # shape as argument
        v1[self.impossible_states] = 0
        v2 = np.random.rand(n_states) # random initialization here
        v2[self.impossible_states] = 0
        delta = np.inf
        num = 0
        while (delta > eps) and num <10000:
            v_old1 = np.copy(v1)
            v_old2 = np.copy(v2)
            for i in range(n_states):
                q1 = np.zeros(n_actions)
                q2 = np.zeros(n_actions)
                for j in self.possible_actions_from_state[i]:
                    next_state = self.env.state_index_transition(i, j)
                    q1[j] = r1_ref[i] + self.time_disc1*v1[next_state]
                    q2[j] = r2_ref[i] + self.time_disc2*v2[next_state]
                actionstar = np.argmax(self.cc_constant * q1 + q2) #returns index of the max value action
                if flip: v1[i] = q1[actionstar]
                else: v2[i] = q2[actionstar]
                flip = not flip
            num = num + 1 # to prevent taking too long
            delta = max(np.max(np.abs(v_old1 - v1)), np.max(np.abs(v_old2 - v2)))
        #print("deterministic value iteration took", num, "steps to converge")
        return v1, v2

    def subjective_reward(self, objective_reward): #TODO default baseline can be overwritten
        if (objective_reward > self.baseline):
            return (objective_reward - self.baseline)**self.alpha
        else:
            return ((-1)*self.kappa*((self.baseline - objective_reward)**self.beta))

    def subjective_probability(self, objective_probability):  # default belief can be overwritten
        return (objective_probability ** self.eta) / (objective_probability ** self.eta + (1 - objective_probability) ** self.eta)