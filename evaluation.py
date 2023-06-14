
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def cosine_similarity_matrix(policy1, policy2):
#     similarity_matrix = cosine_similarity(policy1, policy2)
#     return similarity_matrix


def compute_average_cosine_similarity(policy1, policy2):
    policy1 = policy1.flatten()  # Flatten policy vector 1
    policy2 = policy2.flatten()  # Flatten policy vector 2

    dot_product = np.dot(policy1, policy2)
    norm1 = np.linalg.norm(policy1)
    norm2 = np.linalg.norm(policy2)

    cosine_similarity = dot_product / (norm1 * norm2)

    return np.round(cosine_similarity, 2)  # this is a matrix NO?


#which rewards?
def reward_comparison(expert_trajectories, irl_trajectories, optimal_trajectories, r, r1, r2):
    rewards_expert = 0
    r1_expert = 0
    r2_expert = 0
    for t in expert_trajectories:
        for s in t: #for each visited state
            rewards_expert += r[s]
    rewards_expert = rewards_expert/len(expert_trajectories)

    rewards_irl = 0
    r1_irl = 0
    r2_irl = 0
    for t in irl_trajectories:
        for s in t: #for each visited state
            rewards_irl += r[s]
    rewards_irl = rewards_irl/len(irl_trajectories)

    rewards_optimal = 0
    r1_optimal = 0
    r2_optimal = 0
    for t in optimal_trajectories:
        for s in t:  # for each visited state
            rewards_optimal += r[s]
    rewards_optimal = rewards_optimal / len(optimal_trajectories)

    length = len(optimal_trajectories)#all of them have the same number of trajectories compared for

    rewards_dict = {
        "rewards_expert": np.round(rewards_expert/length, 2),
        "r1_expert": np.round(r1_expert/length, 2),
        "r2_expert": np.round(r2_expert/length, 2),
        "rewards_irl": np.round(rewards_irl/length, 2),
        "r1_irl": np.round(r1_irl/length, 2),
        "r2_irl": np.round(r2_irl/length, 2),
        "rewards_optimal": np.round(rewards_optimal/length, 2),
        "r1_optimal": np.round(r1_optimal/length, 2),
        "r2_optimal": np.round(r2_optimal/length, 2)
    }
    return rewards_dict


def policy_comparison(expert_policy, irl_policy, optimal_policy):
    avg_sim_irl_exp = compute_average_cosine_similarity(irl_policy, expert_policy)
    avg_sim_irl_opt = compute_average_cosine_similarity(irl_policy, optimal_policy)
    avg_sim_exp_opt = compute_average_cosine_similarity(expert_policy, optimal_policy)
    cosine_sim_dict = {
        "avg_sim_irl_exp": avg_sim_irl_exp,
        "avg_sim_irl_opt": avg_sim_irl_opt,
        "avg_sim_exp_opt": avg_sim_exp_opt
    }
    return cosine_sim_dict



#convergence test?

