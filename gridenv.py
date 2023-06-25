import numpy as np


class GridEnvironment():
    def __init__(self, settings):
        self.settings = settings
        self.deterministic = settings["Environment"]["deterministic"]

        # generic characteristics
        self.width = settings["Environment"]["width"]
        self.height = settings["Environment"]["height"]
        self.n_states = self.size = self.width * self.height
        self.actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.n_actions = len(self.actions)


        #rewards and probabilities
        self.r1 = self.r2 = None
        self.r = None
        self.p1 = None
        self.rp_1 = None

        self.road_indices = []
        self.impossible_states = []
        if settings["Environment"]["roads"]["road_length"] != 0:
            roads_horizontal = settings["Environment"]["roads"]["horizontal"] # this is a list of indices
            roads_vertical = settings["Environment"]["roads"]["vertical"]
            states = list(range(0, self.n_states))
            self.road_indices = list(set(roads_vertical + roads_horizontal))
            self.impossible_states = [x for x in states if x not in self.road_indices]

        self.p_transition = self._transition_prob_table()  # of form table[s_from, s_to, a] shape num_states, num_states, num_actions
        self.possible_actions_from_state = self.possible_actions_table()


    def possible_actions_table(self):
        table = [[] for _ in range(self.n_states)] # list of lists, state index here is s_from
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if not np.argmax(self.p_transition[s, :, a]) == s:
                    # if it's equal to itself, it means this action is not possible from that state
                    table[s].append(a)
        return table

    def set_objective_r1_and_r2(self, r1, r2, p1):  #p1 is just ones if deterministic
        self.r1 = r1
        self.r2 = r2
        self.p1 = p1
        self.rp_1 = np.multiply(r1, p1)
        #print("COPY")
        #print(self.rp_1)
        self.r = np.add(self.rp_1, r2)
        return {
            "n_states": self.n_states,
            "num road cells": len(self.road_indices),
            "ratio of usable states": len(self.road_indices)/self.n_states,
            "sum_r1": np.sum(self.rp_1),
            "sum_r2": np.sum(r2),
            "sum_positive_rewards": np.sum(self.r[self.r > 0]),
            "sum_negative_rewards": np.sum(self.r[self.r < 0]),
            "sum_r": np.sum(self.r),
            "simple_rp_1": self.rp_1
        }


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
        if s in self.impossible_states:
            return s
        x, y = self.state_index_to_point(s)
        dx, dy = self.actions[a]
        x += dx
        y += dy
        resulting_index = self.state_point_to_index_clipped((x, y))
        if resulting_index in self.impossible_states:
            return s
        else:
            return resulting_index

    def state_features_one_dim(self):
        return np.identity(self.n_states)

    def _transition_prob_table(self): # of form table[s_from, s_to, a]
        '''
        for all impossible states, all actions lead to itself
        '''
        table = np.zeros(shape=(self.n_states, self.n_states, self.n_actions))

        for s_from in range(self.n_states):
            for s_to in range(self.n_states):
                for a in range(self.n_actions):
                    table[s_from, s_to, a] = self._transition_prob(s_from, s_to, a)

        return table

    def within_grid_bool(self, x, y):
        #return not 0 <= x < self.width or not 0 <= y < self.height  # these are valid states
        return 0 <= x < self.width and 0 <= y < self.height  # state is within dimensions?

    def index_within_grid_bool(self, state_index):
        x, y = self.state_index_to_point(state_index)
        return 0 <= x < self.width and 0 <= y < self.height  # state is within dimensions?

    def _transition_prob(self, s_from, s_to, a):
        '''
        in below cases, action leads to same state:
            - if out of bounds (<0 or >widhth or >height)
            - s_from is a road and s_to is not
        action leads from s_from to s_to taking action only if  x_new == tx and y_new == ty
        '''
        if not self.index_within_grid_bool(s_from) or not self.index_within_grid_bool(s_to): # if either is not a valid state
            return 0.0
        #  at this point we know both s_from and s_to are inside grid

        if s_from == s_to and s_from in self.impossible_states:  # impossible states lead to themselves for every action
            return 1.0

        if s_to in self.impossible_states or s_from in self.impossible_states:
            # if s_to is an impossible state it can't be reached by other states
            # OR
            # if s_from is impossible, it is not possible to transition anywhere
            return 0.0

        #  at this point we know s_to and s_from are both possible,
        #  see if resulting state fits with given action and is possible

        fx, fy = self.state_index_to_point(s_from)
        tx, ty = self.state_index_to_point(s_to)
        ax, ay = self.actions[a]

        x_new = fx + ax
        y_new = fy + ay

        resulting_state = self.state_point_to_index((x_new, y_new))

        if resulting_state in self.impossible_states or not self.index_within_grid_bool(resulting_state):
            if s_to == s_from:
                return 1.0
            else: return 0.

        # from now we know resulting state is valid
        if x_new == tx and y_new == ty: #action leads to
            return 1.0
        else:
            return 0.0

    def __repr__(self):
        return "GridWorld(size={})".format(self.size)

    #def _transition_prob_table(self):



