
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

    return cosine_similarity  # this is a matrix NO?


#which rewards?
def trajectory_comparison(expert_trajectories, irl_trajectories, r, reward_comparison=True): #because reward I just add up
    rewards_expert = 0
    for t in expert_trajectories:
        for s in t: #for each visited state
            if reward_comparison: rewards_expert += r[s]
    rewards_expert = rewards_expert/len(expert_trajectories)

    rewards_irl = 0
    for t in irl_trajectories:
        for s in t: #for each visited state
            if reward_comparison: rewards_irl += r[s]
    rewards_irl = rewards_irl/len(irl_trajectories)

    return rewards_expert, rewards_irl


def policy_comparison(expert_policy, irl_policy):
    sim_array = cosine_similarity(expert_policy, irl_policy)
    print("sim_array.shape", sim_array.shape)
    sim_array_states = np.mean(sim_array, axis=1)
    avg_sim = compute_average_cosine_similarity(expert_policy, irl_policy)
    return sim_array_states, avg_sim



#convergence test?

