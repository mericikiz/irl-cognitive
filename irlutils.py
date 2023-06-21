import numpy as np
from rp1.gridenv import GridEnvironment
import random
from itertools import chain
from itertools import product


action_numbers = np.array([0, 1, 2, 3]) #fix this to be more adaptable TODO
generic = 0

def softmax(x, temperature):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / np.sum(e_x)

def erase_loops(route, semi_target):
    if (len(semi_target) == 0):
        return one_path(route)
    else:
        arr_all = route
        mask = np.isin(arr_all, semi_target)
        first_occurrence = np.argmax(mask)
        segment1 = arr_all[:first_occurrence + 1]
        segment2 = arr_all[first_occurrence + 1:]
        result = np.concatenate((one_path(segment1), one_path(segment2)))
        result = result.astype(int)
        return result


def multiple_visits_possible(route, semi_target):
    arr_all = np.copy(route)
    extracted_segments = []
    while True:
        mask = np.isin(route, semi_target)
        if not np.any(mask):  # if none of the elements are present anymore
            extracted_segments.append(arr_all[0:])  # add the last bit
            break
        first_occurrence = np.argmax(mask)
        segment = arr_all[:first_occurrence + 1]
        extracted_segments.append(segment)
        arr_all = arr_all[first_occurrence + 1:]
    result = np.array([])
    for seg in extracted_segments:
        result = np.concatenate((result, one_path(seg)))
    print("done with hard path editing for one path")
    return result


def one_path(path): #only allowed to visit a state once
    working_array = path
    uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
    duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
    while not (len(duplicate_elements) == 0):
        element = duplicate_elements[0]
        duplicates = (np.where(element == working_array))  # indices of the duplicates
        beginning_to_first_occurrence = working_array[0:np.amin(duplicates)]
        last_occurrence_to_end = working_array[np.amax(duplicates):len(working_array)]
        working_array = np.concatenate((beginning_to_first_occurrence, last_occurrence_to_end))
        uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
        duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
    return working_array


def stochastic_policy_arr(value_iteration_array, env, just_value, w):
    # value_iteration_array can also be q table
    # in previous code version: policy = S.stochastic_policy_from_value(self.env, value_final, w=weighting)
    # TODO what is this policy_array doing here??
    policy_array = np.zeros((env.n_states, env.n_actions)) #every action for every state will have probability associated
    if just_value:
        policy = np.array([
            np.array([w(value_iteration_array[env.state_index_transition(s, a)]) for a in range(env.n_actions)])
            for s in range(env.n_states)
        ])
        policy = np.round(policy, 2)
        print("policy", policy.shape, policy)
        #print("policy_array", policy_array.shape, policy_array)
        # Apply division to specific states, making indices that require division by zero 0
        # every row except impossible states should sum to 1
        denominator = np.sum(policy, axis=1)[:, None]
        print("denominator", denominator)
        result = np.zeros_like(policy)
        result = np.divide(policy, denominator, where=denominator != 0, out=result)
        #result = np.where(denominator != 0, np.divide(policy, denominator), 0)
        print("result", result)
        print("result row sums", np.sum(result, axis=1))
        #result[np.isnan(result)] = None
        return result

        #return policy / np.sum(policy, axis=1)[:, None]

    else:
        return stochastic_policy_arr_from_q_value(value_iteration_array)


def stochastic_policy_arr_from_q_value(q_table):
    print("stochastic_policy_arr_from_q_value not yet implemented") # TODO, can be extended
    pass


def states(trajectory):
    """ '(state_from, action, state_to)` to states """
    return map(lambda x: x[0], chain(trajectory, [(trajectory[-1][2], 0, 0)]))

def generate_trajectory_gridworld(env, start, final, semi_target, policy_array):
    """
    Generate a single trajectory.

    Args:
        start: starting state index.
        final: collection of terminal states where trajectory ends

    All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
    """

    state = int(start)

    trajectory = []
    steps = 0
    while state not in final: # or num < 200: #state=current state
        value_actions = policy_array[state, :].copy() #includes None s
        value_actions[np.where(value_actions == None)] = 0 #TODO really?

        action = np.random.choice(action_numbers, p=value_actions) #A function (state: Integer) -> (action: Integer) mapping a state to an action

        next_state = int(env.state_index_transition(state, action))
        trajectory += [(state, action, next_state)]
        state = int(next_state)
        steps += 1
        #if len(trajectory)>3*env.n_states: return None # TODO instead improve your trajectory algorithm
    print("generated 1 trajectory in", steps, "steps, trajectory length ", len(trajectory))
    return trajectory #transitions, array of tuples in  form `(state_from, action, state_to)`

def generate_trajectory_gridworld_OLD(env, start, final, semi_target, policy_array):
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    """
    Generate a single trajectory.

    Args:
        env: The world for which the trajectory should be generated.
        start: The starting state (as Integer index).
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            returned.

    All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
    """

    state = int(start)

    trajectory = []
    num = 0  # TODO find a better solution, maybe a filter after a trajectory ends?
    steps = 0
    init_probability = 0.1
    #next_s = range(env.n_states)
    prev_state=int(1000)
    while state not in final: # or num < 200: #state=current state
        if num >100: probability = 0.01
        else: probability = 0.8
        #if steps%1000==0: init_probability=init_probability*1.5
        #next_state = state
        value_actions = policy_array[state, :].copy()
        #print("state is", state, "probabilities", value_actions)
        if random.random() < probability:
            action = random.choice(action_numbers)
            num = 0
        else:
            action = np.random.choice(action_numbers, p=value_actions) #A function (state: Integer) -> (action: Integer) mapping a state to an action
            num = num+1
        #num = num + 1
        #next_p = env.p_transition[state, :, action]
        #next_state = np.random.choice(next_s, p=next_p)
        possible_next_state = int(env.state_index_transition(state, action))
        if possible_next_state == prev_state and prev_state not in semi_target:
            avoid_action = action
            value_actions[avoid_action] = 0.0
            value_actions = value_actions/np.sum(value_actions)
            action = np.random.choice(action_numbers, p=value_actions)
            next_state = int(env.state_index_transition(state, action))
            num = num + 1
        else:
            next_state = int(possible_next_state)
        trajectory += [(state, action, next_state)]
        prev_state = int(state)
        state = int(next_state)
        steps += 1
        #if len(trajectory)>3*env.n_states: return None # TODO instead improve your trajectory algorithm
    #print("generated 1 trajectory in", steps, "steps, trajectory length ", len(trajectory))
    return trajectory #transitions, array of tuples in  form `(state_from, action, state_to)`


def generate_trajectories_gridworld(n, env, start, final, semi_target, policy_array): #TODO
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    # usage in my previous version: tjs = list(T.generate_trajectories(n_trajectories, self.world, policy_exec, self.start, self.terminal))
    """
    Generate multiple trajectories.

    Args:
        n: The number of trajectories to generate.
        world: The world for which the trajectories should be generated.
        policy_execution: A function `(state: Integer) -> action: Integer` mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index), a list of starting
            states (with uniform probability), or a list of starting state
            probabilities, mapping each state to a probability. Iff the
            length of the provided list is equal to the number of states, it
            is assumed to be a probability distribution over all states.
            Otherwise it is assumed to be a list containing all starting
            state indices, an individual state is then chosen uniformly.
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            complete.

    Returns: `n` `Trajectory` lists
    """

    s = np.random.choice(start)
    generated = 0
    print("generating trajectories")
    trajlist = []
    while generated < n:
        traj = generate_trajectory_gridworld(env, s, final, semi_target, policy_array)
        if traj != None:
            trajlist.append(traj)
            generated+=1

    return trajlist


def feature_expectation_from_trajectories(features, trajectory_states, eliminate_loops):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)
    if eliminate_loops: # TODO maybe treat differently when loops are eliminated
        for t in trajectory_states:
            for s in t:
                fe += features[s, :]
    else:  # for now it's the same, can be extended
        for t in trajectory_states:
            for s in t:
                fe += features[s, :]
        # for t in trajectories:                  # for each trajectory
        #     for s in t.states():                # for each state in trajectory
        #         fe += features[s, :]            # sum-up features

    return fe / len(trajectory_states)           # average over trajectories

def initial_probabilities_from_trajectories(n_states, trajectory_states, eliminate_loops):
    p = np.zeros(n_states)

    if eliminate_loops: # meaning loops are already eliminated and trajectories is in list of lists form, just states
        for t in trajectory_states:  # for each trajectory
            p[t[0]] += 1.0  # increment starting state # TODO treat differently when loops eliminated?
    else:
        for t in trajectory_states:                  # for each trajectory
            p[t[0]] += 1.0  # increment starting state
            # p[t.transitions()[0][0]] += 1.0     # increment starting state

    return p / len(trajectory_states)            # normalize


def compute_expected_svf(p_transition, p_initial, terminal, reward, impossible_states, eps=1e-5):
    n_states, _, n_actions = p_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states
    #is_obstacle = np.zeros(n_states, dtype=bool)
    #is_obstacle[impossible_states] = True

    # Backward Pass
    # 1. initialize at terminal states
    zs = np.zeros(n_states)  # zs: state partition function
    zs[terminal] = 1.0

    # 2. perform backward pass
    for _ in range(2 * n_states):  # longest trajectory: n_states
        # reset action values to zero
        za = np.zeros((n_states, n_actions))  # za: action partition function
        # for each state-action pair
        for s_from, a in product(range(n_states), range(n_actions)):
            # sum over s_to
            for s_to in range(n_states):
                za[s_from, a] += p_transition[s_from, s_to, a] * np.exp(reward[s_from]) * zs[s_to]

        # sum over all actions
        zs = za.sum(axis=1)

    # 3. compute local action probabilities
    p_action = za / zs[:, None]

    # Forward Pass
    # 4. initialize with starting probability
    d = np.zeros((n_states, 2 * n_states))  # d: state-visitation frequencies
    d[:, 0] = p_initial

    # 5. iterate for N steps
    for t in range(1, 2 * n_states):  # longest trajectory: n_states

        # for all states
        for s_to in range(n_states):

            # sum over nonterminal state-action pairs
            for s_from, a in product(nonterminal, range(n_actions)):
                d[s_to, t] += d[s_from, t - 1] * p_action[s_from, a] * p_transition[s_from, s_to, a]

    # 6. sum-up frequencies
    return d.sum(axis=1)


def maxent_irl(p_transition, features, terminal, trajectory_states, optim, init, eliminate_loops, impossible_states, eps=1e-4):
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectory_states, eliminate_loops)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectory_states, eliminate_loops)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward, impossible_states)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    #TODO not so sure about e_svf being the most recent updated one, so do again
    reward_maxent = features.dot(omega)
    #e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward_maxent, impossible_states)
    return reward_maxent, p_initial, e_svf, e_features





# Note: this code will only work with one feature per state
# p_initial = irlutils.initial_probabilities_from_trajectories(self.env.n_states, expert_trajectories, self.eliminate_loops)
# e_svf = irlutils.compute_expected_svf(self.env.p_transition, p_initial, self.settings["IRL"]["terminal"], reward_maxent)
# e_features = irlutils.feature_expectation_from_trajectories(features, expert_trajectories, self.eliminate_loops)