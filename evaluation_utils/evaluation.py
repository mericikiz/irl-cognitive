
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def cosine_similarity_matrix(policy1, policy2):
#     similarity_matrix = cosine_similarity(policy1, policy2)
#     return similarity_matrix


def compute_average_cosine_similarity(policy1, policy2):
    policy1_flat = policy1.flatten()  # Flatten policy vector 1
    policy2_flat = policy2.flatten()  # Flatten policy vector 2

    dot_product = np.dot(policy1_flat, policy2_flat)
    norm1 = np.linalg.norm(policy1_flat)
    norm2 = np.linalg.norm(policy2_flat)

    cosine_similarity = dot_product / (norm1 * norm2)

    return np.round(cosine_similarity, 2)


def collect_all_rewards(trajectories, r1, r2):
    r1_sum = 0
    r2_sum = 0
    for t in trajectories:
        for s in t:  # for each visited state
            r1_sum += r1[s]
            r2_sum += r2[s]
    r1_avg = r1_sum / len(trajectories)
    r2_avg = r2_sum / len(trajectories)
    r_avg = r1_avg + r2_avg

    return r_avg, r1_avg, r2_avg


def reward_comparison(expert_trajectories, irl_trajectories, optimal_trajectories, r, r1, r2):
    # r and r1 take uncertainty into account and do expected average
    rewards_expert, r1_expert, r2_expert = collect_all_rewards(expert_trajectories, r1, r2)
    rewards_irl, r1_irl, r2_irl = collect_all_rewards(irl_trajectories, r1, r2)
    rewards_optimal, r1_optimal, r2_optimal = collect_all_rewards(optimal_trajectories, r1, r2)
    # ^^^ these are average over many generated trajectories

    # compare max and mins?
    #length = len(optimal_trajectories)  # all of them have the same number of trajectories compared for

    rewards_dict = {
        "rewards_expert": np.round(rewards_expert, 2),
        "r1_expert": np.round(r1_expert, 2),
        "r2_expert": np.round(r2_expert, 2),
        "rewards_irl": np.round(rewards_irl, 2),
        "r1_irl": np.round(r1_irl, 2),
        "r2_irl": np.round(r2_irl, 2),
        "rewards_optimal": np.round(rewards_optimal, 2),
        "r1_optimal": np.round(r1_optimal, 2),
        "r2_optimal": np.round(r2_optimal, 2)
    }
    return rewards_dict


def policy_comparison(expert_policy, irl_policy, optimal_policy):
    # TODO!! check if pass by reference messes up some things
    avg_sim_irl_exp = compute_average_cosine_similarity(irl_policy, expert_policy)
    avg_sim_irl_opt = compute_average_cosine_similarity(irl_policy, optimal_policy)
    avg_sim_exp_opt = compute_average_cosine_similarity(expert_policy, optimal_policy)
    cosine_sim_dict = {
        "avg_sim_irl_exp": avg_sim_irl_exp,
        "avg_sim_irl_opt": avg_sim_irl_opt,
        "avg_sim_exp_opt": avg_sim_exp_opt,
    }
    return cosine_sim_dict

# def compare_policies_by_cell(expert_policy, irl_policy):
#     states_from, states_to, actions = expert_policy.shape
# 
#     # Initialize a similarity matrix
#     similarity_matrix = np.zeros(states_from)
#
#     # Compare policies cell by cell
#     for row in range(n_rows):
#         for col in range(n_cols):
#             if expert_policy[row, col] == irl_policy[row, col]:
#                 similarity_matrix[row, col] = 1
#
#     return similarity_matrix
#
# #convergence test?

