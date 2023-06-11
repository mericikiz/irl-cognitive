import numpy as np
from rp1.gridenv import GridEnvironment
import random
from itertools import chain
from itertools import product


action_numbers = [0, 1, 2, 3] #fix this to be more adaptable TODO
generic = 0
def softmax(x, temperature):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / np.sum(e_x)

def erase_loops(route):
    working_array = route
    uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
    duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
    while not (len(duplicate_elements) == 0):
        element = duplicate_elements[0]
        duplicates = (np.where(element == working_array)) #indices of the duplicates
        beginning_to_first_occurrence = working_array[0:np.amin(duplicates)]
        last_occurrence_to_end = working_array[np.amax(duplicates):len(working_array)]
        working_array = np.concatenate((beginning_to_first_occurrence, last_occurrence_to_end))
        uniqueValues, indicesList = np.unique(working_array, return_inverse=True)
        duplicate_elements = uniqueValues[np.bincount(indicesList) > 1]
    return working_array


def stochastic_policy_arr(value_iteration_array, env, just_value, w):
    # value_iteration_array can also be q table
    # in previous code version: policy = S.stochastic_policy_from_value(self.env, value_final, w=weighting)
    policy_array = np.zeros((env.n_states, env.n_actions)) #every action for every state will have probability associated
    if just_value:
        policy = np.array([
            np.array([w(value_iteration_array[env.state_index_transition(s, a)]) for a in range(env.n_actions)])
            for s in range(env.n_states)
        ])
        return policy / np.sum(policy, axis=1)[:, None]
        #for s in range(env.n_states):
        #    values = np.array([w(value_iteration_array[env.state_index_transition(s, a)]) for a in range(env.n_actions)])
        #    probabilities = softmax(values, temperature)  # Apply softmax with temperature
        #    policy_array[s] = probabilities
        #return policy_array #every row should sum to 1
    else:
        return stochastic_policy_arr_from_q_value(value_iteration_array)

def stochastic_policy_arr_from_q_value(q_table):
    print("stochastic_policy_arr_from_q_value not yet implemented") #TODO
    pass

def states(trajectory):
    """ '(state_from, action, state_to)` to states """
    return map(lambda x: x[0], chain(trajectory, [(trajectory[-1][2], 0, 0)]))

def generate_trajectory_gridworld(env, policy_execution, start, final):
    # BORROWED AND MODIFIED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    """
    Generate a single trajectory.

    Args:
        world: The world for which the trajectory should be generated.
        policy_execution: A function (state: Integer) -> (action: Integer) mapping a
            state to an action, specifying which action to take in which
            state. This function may return different actions for multiple
            invokations with the same state, i.e. it may make a
            probabilistic decision and will be invoked anew every time a
            (new or old) state is visited (again).
        start: The starting state (as Integer index).
        final: A collection of terminal states. If a trajectory reaches a
            terminal state, generation is complete and the trajectory is
            returned.

    All transitions in this trajectory as array of tuples
            `(state_from, action, state_to)`.
    """

    state = start

    trajectory = []
    num = 0  # TODO find a better solution, maybe a filter after a trajectory ends?
    steps = 0
    probability = 0.1
    prev_state = state
    while state not in final: # or num < 200:
        #if num >100: probability = 0.7
        #else: probability = 0.1
        # if random.random() < probability:
        #     action = random.choice(action_numbers)
        #     num = 0
        #     #probability = probability * 5
        # else:
        #     action = policy_execution(state)
        next_state = state
        next_s = range(env.n_states)

        while (next_state==prev_state): #TODO improve, current solution is if next state is equal to previous, choose again
            #action = random.choice(action_numbers)
            action = policy_execution(state)
            next_p = env.p_transition[state, :, action]
            next_state = np.random.choice(next_s, p=next_p)

        trajectory += [(state, action, next_state)]
        prev_state = state
        state = next_state
        #num = num + 1
        steps += 1
        #if len(trajectory)>3*env.n_states: return None # TODO instead improve your trajectory algorithm
    #print("generated 1 trajectory in", steps, "steps, trajectory length ", len(trajectory))
    #print(list(states(trajectory)))
    return trajectory #transitions, array of tuples in  form `(state_from, action, state_to)`
    #if num < 200:
    #    print("generated 1 trajectory")
    #    return Trajectory(trajectory)
    #else: return None


def generate_trajectories_gridworld(n, env, policy_execution, start, final, eliminate_loops): #TODO
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
    print("generating expert trajectories")
    trajlist = []
    while generated < n:
        traj = generate_trajectory_gridworld(env, policy_execution, s, final)
        if traj != None:
            trajlist.append(traj)
            generated+=1

    return trajlist

def stochastic_policy_adapter(policy):
    #BORROWED FROM GITHUB https://github.com/qzed/irl-maxent/tree/master
    """
    A policy adapter for stochastic policies.

    Adapts a stochastic policy given as array or map
    `policy[state, action] -> probability` for the trajectory-generation
    functions.

    Args:
        policy: The stochastic policy as map/array
            `policy[state: Integer, action: Integer] -> probability`
            representing the probability distribution p(action | state) of
            an action given a state.

    Returns:
        A function `(state: Integer) -> action: Integer` acting out the
        given policy, choosing an action randomly based on the distribution
        defined by the given policy.
    """
    return lambda state: np.random.choice([*range(policy.shape[1])], p=policy[state, :])

def feature_expectation_from_trajectories(features, trajectories):
    n_states, n_features = features.shape

    fe = np.zeros(n_features)

    for t in trajectories:                  # for each trajectory
        for s in t.states():                # for each state in trajectory
            fe += features[s, :]            # sum-up features

    return fe / len(trajectories)           # average over trajectories

def initial_probabilities_from_trajectories(n_states, trajectories, eliminate_loops):
    p = np.zeros(n_states)

    if eliminate_loops: # meaning loops are already eliminated and trajectories is in list of lists form, just states
        for t in trajectories:  # for each trajectory
            p[t[0]] += 1.0  # increment starting state
    else:
        for t in trajectories:                  # for each trajectory
            p[t.transitions()[0][0]] += 1.0     # increment starting state

    return p / len(trajectories)            # normalize


def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    n_states, _, n_actions = p_transition.shape
    nonterminal = set(range(n_states)) - set(terminal)  # nonterminal states

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


def maxent_irl(p_transition, features, terminal, trajectories, optim, init, eliminate_loops, eps=1e-4):
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute feature expectation from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories, eliminate_loops)

    # compute starting-state probabilities from trajectories
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories, eliminate_loops)

    # gradient descent optimization
    omega = init(n_features)  # initialize our parameters
    delta = np.inf  # initialize delta for convergence check

    optim.reset(omega)  # re-start optimizer
    while delta > eps:  # iterate until convergence
        omega_old = omega.copy()

        # compute per-state reward from features
        reward = features.dot(omega)

        # compute gradient of the log-likelihood
        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward)
        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)

        # re-compute detla for convergence check
        delta = np.max(np.abs(omega_old - omega))

    # re-compute per-state reward and return
    return features.dot(omega)