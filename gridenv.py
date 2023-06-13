import numpy as np


class GridEnvironment():
    def __init__(self, name, width, height, deterministic, vis_dict):

        self.name = name
        self.deterministic = deterministic

        # generic characteristics
        self.n_states = self.size = width * height
        self.width = width
        self.height = height
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.n_actions = len(self.actions)

        self.p_transition = self._transition_prob_table()

        #rewards and probabilities
        self.r1 = self.r2 = None
        self.r = None
        self.p1 = None
        self.rp_1 = None

        self.visual_dict = vis_dict

    def set_objective_r1_and_r2(self, r1, r2, p1):  #p1 is just ones if deterministic
        self.r1 = r1
        self.r2 = r2
        self.p1 = p1
        self.rp_1 = np.multiply(r1, p1)
        self.r = np.add(self.rp_1, r2)

    def state_index_to_point(self, state):
        x = state % self.width
        y = state // self.width
        return x, y

    def state_point_to_index(self, state):
        x, y = state
        return y * self.width + x

    def state_point_to_index_clipped(self, state):
        x, y = state
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return self.state_point_to_index((x, y))

    def state_index_transition(self, s, a):
        x, y = self.state_index_to_point(s)
        dx, dy = self.actions[a]
        x += dx
        y += dy
        return self.state_point_to_index_clipped((x, y))

    def state_features_one_dim(self):
        return np.identity(self.n_states)

    def _transition_prob_table(self): # of form table[s_from, s_to, a]
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        for s_from in range(self.n_states):
            for s_to in range(self.n_states):
                for a in range(self.n_actions):
                    table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def _transition_prob(self, s_from, s_to, a):
        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        if fx + ax == tx and fy + ay == ty:
            return 1.0

        if fx == tx and fy == ty:
            if not 0 <= fx + ax < self.width or not 0 <= fy + ay < self.height:
                return 1.0

        return 0.0 # otherwise impossible transition given

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)

    #def _transition_prob_table(self):



