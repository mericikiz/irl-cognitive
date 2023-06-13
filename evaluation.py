
import numpy as np

def cosine_similarity(policy1, policy2):
    policy1 = policy1.flatten()  # Flatten policy vector 1
    policy2 = policy2.flatten()  # Flatten policy vector 2

    dot_product = np.dot(policy1, policy2)
    norm1 = np.linalg.norm(policy1)
    norm2 = np.linalg.norm(policy2)

    cosine_similarity = dot_product / (norm1 * norm2)

    return cosine_similarity

def compute_average_cosine_similarity(similarity_matrix):
    average_similarity = np.mean(similarity_matrix)
    return average_similarity


#which rewards?
def trajectory_comparison(expert_trajectories, irl_trajectories, reward_comparison=True): #because reward I just add up
    pass

def policy_comparison(expert_policy, irl_policy):
    pass


#convergence test?

