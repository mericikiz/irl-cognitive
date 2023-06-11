import numpy as np
from rp1.helpers import solver_modified as S

class Cognitive_model():

    def __init__(self, env, alpha, beta, kappa, eta, gamma1, gamma2, cc_constant):

        # COGNITIVE PARAMETERS
        self.cc_constant = cc_constant # linear term
        self.cc_cost = lambda x: x*self.cc_constant #cc_constant is cognitive_control_cost function, linear for now
        self.time_disc1 = gamma1
        self.time_disc2 = gamma2
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.eta = eta
        # baseline reward and baseline probability depend on individual calculation

        # ENVIRONMENT, REWARDS AND VALUES
        self.env = env
        # TODO env.p_transition
        # p_transition exists regardless of env being deterministic or not, if deterministic each transition simply has a definitive outcome

        self.reward_arr1_o = np.copy(self.env.r1) #o stands for objective
        self.v1_o = S.value_iteration(self.env.p_transition, self.reward_arr1_o, discount=self.time_disc1)
        self.v1_comb_o = None

        self.reward_arr2_o = np.copy(self.env.r2)
        self.v2_o = S.value_iteration(self.env.p_transition, self.reward_arr2_o, discount=self.time_disc2)
        self.v2_comb_o = None

        self.value_it_1_and_2_soph_o = None

        self.combine_value_iteration()

        if self.env.tests_dict["test normalization"]:
            #normalized test, just to see the difference
            self.v1_o_n = None
            self.v2_o_n = None
            self.v1_comb_o_n = None
            self.v2_comb_o_n = None
            self.value_it_1_and_2_soph_o_n = None
            self.normalized_test()

        if self.env.tests_dict["test subjective valuation"]:
            self.r1_subj_v = self.r2_subj_v = None
            self.v1_subj_v = self.v2_subj_v = None
            self.v1_comb_subj = self.v2_comb_subj = None
            self.value_it_1_and_2_soph_subj = None
            self.subj_valuation()

        if self.env.tests_dict["test subjective probability"]:
            self.r1_sub_p = None
            self.r2_sub_p = None



    def combine_value_iteration(self):
        if self.env.deterministic:
            v1, v2 = self.combine_value_iteration_deterministic(self.reward_arr1_o, self.reward_arr2_o)
            self.value_it_1_and_2_soph_o = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_o
            self.v1_comb_o = v1
            self.v2_comb_o = v2
        else: self.combine_value_iteration_uncertainty()

    def normalized_test(self):
        self.v1_o_n = S.value_iteration(self.env.p_transition, self.env.r1_n, discount=self.time_disc1)
        self.v2_o_n = S.value_iteration(self.env.p_transition, self.env.r2_n, discount=self.time_disc2)
        v1, v2 = self.combine_value_iteration_deterministic(self.env.r1_n, self.env.r2_n)
        self.value_it_1_and_2_soph_o_n = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_o_n
        self.v1_comb_o_n = v1
        self.v2_comb_o_n = v2

    def subj_valuation(self, baseline=0):
        vectorized_func = np.vectorize(self.subjective_reward)
        self.r1_subj_v = vectorized_func(self.reward_arr1_o, baseline)
        self.r2_subj_v = vectorized_func(self.reward_arr2_o, baseline)
        self.v1_subj_v = S.value_iteration(self.env.p_transition, self.r1_subj_v, discount=self.time_disc1)
        self.v2_subj_v = S.value_iteration(self.env.p_transition, self.r2_subj_v, discount=self.time_disc2)
        v1, v2 = self.combine_value_iteration_deterministic(self.r1_subj_v, self.r2_subj_v)
        self.value_it_1_and_2_soph_subj = self.cc_constant * v1 + v2 - self.cc_constant * self.v1_subj_v
        self.v1_comb_subj = v1
        self.v2_comb_subj = v2

    def combine_value_iteration_deterministic(self, r1_ref, r2_ref, eps=1e-5):
        n_states = self.env.n_states
        n_actions = self.env.n_actions
        flip = True # interleave the update of two value iterations
        v1 = np.random.rand(n_states) # shape as argument
        v2 = np.random.rand(n_states) # random initialization here
        delta = np.inf
        num = 0
        while (delta > eps) and num <100000:
            v_old1 = np.copy(v1)
            v_old2 = np.copy(v2)
            for i in range(n_states):
                q1 = np.zeros(n_actions)
                q2 = np.zeros(n_actions)
                for j in range(n_actions):
                    next_state = self.env.state_index_transition(i, j) # TODO env.state_index_transition(i, j)
                    q1[j] = r1_ref[i] + self.time_disc1*v1[next_state]
                    q2[j] = r2_ref[i] + self.time_disc2*v2[next_state]
                actionstar = np.argmax(self.cc_constant * q1 + q2) #returns index of the max value action
                if flip: v1[i] = q1[actionstar]
                else: v2[i] = q2[actionstar]
                flip = not flip
            num = num + 1 # to prevent taking too long
            delta = max(np.max(np.abs(v_old1 - v1)), np.max(np.abs(v_old2 - v2)))
        print("deterministic value iteration took", num, "steps to converge")
        return v1, v2

    def combine_value_iteration_uncertainty(self):
        print("uncertainty involving value iteration not yet done") # TODO

    def subjective_reward(self, objective_reward, baseline=0): #default baseline can be overwritten
        if (objective_reward > baseline):
            return (objective_reward - baseline)**self.alpha
        else:
            return ((-1)*self.kappa*((baseline - objective_reward)**self.beta))

    def subjective_probability(self, objective_probability, belief = 0.5): #default belief can be overwritten
        return (objective_probability/(objective_probability + (1 - belief)))**self.eta


'''
α (for gains):
α > 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
α = 1: Represents a linear value function, where the subjective value is directly proportional to the gain.
α < 1: Indicates a convex value function, suggesting increasing sensitivity to gains.

β (for losses):
β > 1: Represents a convex value function, indicating increasing sensitivity to losses.
β = 1: Reflects a linear value function, where the subjective value is directly proportional to the loss.
β < 1: Indicates a concave value function, suggesting diminishing sensitivity to losses.

κ (for losses):
κ > 1: Represents the degree of loss aversion, with higher values indicating stronger aversion to losses.
κ = 1: Implies no loss aversion, resulting in a linear weighting of losses.

η (for probability weighting):
η > 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
η = 1: Represents a linear weighting function, where probabilities are evaluated according to their actual values.
η < 1: Indicates an underweighting of small probabilities and overweighting of large probabilities.
'''