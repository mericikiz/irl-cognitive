"""
Generic solver methods for Markov Decision Processes (MDPs) and methods for
policy computations for GridWorld.
"""

import numpy as np
import rp1.gridenv

def uncertainty_value_iteration(p, reward, reward_prob, discount, possible_actions_from_state, eps=1e-3):
    n_states, _, n_actions = p.shape
    v = np.zeros(n_states)

    # Convert p to a deterministic transition matrix for each action
    p_deterministic = np.argmax(p, axis=1) #rows are each state, columns in rows are each action

    delta = np.inf
    num = 0
    while delta > eps:
        v_old = v.copy()
        for s in range(n_states):
            q = np.zeros(n_actions)
            for a in possible_actions_from_state[s]:
                q[a] = reward_prob[s] * reward[s] + (discount * v[p_deterministic[s, a]])

            v[s] = np.max(q)

        delta = np.max(np.abs(v_old - v))
        num += 1

    #print("uncertainty_value_iteration completed, num is", num)

    return v


def value_iteration(p, reward, discount, possible_actions_from_state=None, eps=1e-3):
    """
    Basic value-iteration algorithm to solve the given MDP.

    Args:
        p: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The value function as table `[state: Integer] -> value: Float`.
    """
    n_states, _, n_actions = p.shape
    v = np.zeros(n_states)

    # Setup transition probability matrices for easy use with numpy.
    #
    # This is an array of matrices, one matrix per action. Multiplying
    # state-values v(s) with one of these matrices P_a for action a represents
    # the equation
    #     P_a * [ v(s_i) ]_i^T = [ sum_k p(s_k | s_j, a) * v(s_K) ]_j^T
    p = [np.matrix(p[:, :, a]) for a in range(n_actions)]

    delta = np.inf
    while delta > eps:      # iterate until convergence
        v_old = v

        # compute state-action values (note: we actually have Q[a, s] here)
        q = discount * np.array([p[a] @ v for a in range(n_actions)])

        # compute state values
        v = reward + np.max(q, axis=0)[0]

        # compute maximum delta
        delta = np.max(np.abs(v_old - v))

    return v

def optimal_policy(env, reward, discount, eps=1e-3):
    """
    Compute the optimal policy using value-iteration

    Args:
        world: The `GridWorld` instance for which the policy should be
            computed.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`
        discount: The discount (gamma) applied during value-iteration.
        eps: The threshold to be used as convergence criterion. Convergence
            is assumed if the value-function changes less than the threshold
            on all states in a single iteration.

    Returns:
        The optimal (deterministic) policy given the provided arguments as
        table `[state: Integer] -> action: Integer`.

    See also:
        - `value_iteration`
        - `optimal_policy_from_value`
    """
    value = value_iteration(env.p_transition, reward, discount, env.possible_actions_from_state, eps)
    return optimal_policy_from_value(env, value)


def optimal_policy_from_value(env, value):
    """
    Compute the optimal policy from the given value function.

    Args:
        world: The `GridWorld` instance for which the the policy should be
            computed.
        value: The value-function dictating the policy as table
            `[state: Integer] -> value: Float`

    Returns:
        The optimal (deterministic) policy given the provided arguments as
        table `[state: Integer] -> action: Integer`.
    """
    policy = np.array([
        np.argmax([value[env.state_index_transition(s, a)] for a in range(env.n_actions)])
        for s in range(env.n_states)
    ])

    return policy
