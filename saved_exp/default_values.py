import numpy as np


def get_new_cog_dict(alpha=0.7, beta=0.55, kappa=1.5, eta=0.8, time_disc_1=0.9, time_disc_2=0.9, cc_constant=1.0, baseline=0, baseline_changes=False):
    return {
        "cognitive_control": cc_constant,
        "time_disc_1": time_disc_1,
        "time_disc_2": time_disc_2,
        "alpha": alpha,
        "beta": beta,
        "kappa": kappa,
        "eta": eta,
        "baseline": baseline,
        "baseline_changes": baseline_changes
    }

def get_new_exp_info_dict(exp_name="Default", RL_algorithm="Value Iteration", what="Default", mode="subjective", trial_no=99999):
    return {
        "exp name": exp_name,
        "RL algorithm used": RL_algorithm,
        "what is being tested": what,
        "mode": mode,
        "trial no": trial_no
    }

def get_other_params_dict(policy_weighting = lambda x: x**20 , number_of_expert_trajectories = 50, eliminate_loops = True):
# policy_weighting is down-weighting of less optimal actions, how much more likely to take more optimal actions at a given time
    return {
                "policy weighting" : policy_weighting,
                "number of expert trajectories": number_of_expert_trajectories,
                "eliminate loops in trajectory": eliminate_loops
            }

def print_text_env(width, height):
    print("gridworld indices in text")
    np.set_printoptions(linewidth=80, edgeitems=width, suppress=True)
    arr = np.arange(0, width*height).reshape((height, width))

    print(arr[::-1]) #reversed order

def get_settings(populate_with_defaults=True, exp_info_dict=None, cog_param_dict=None, other_param_dict=None):
    if populate_with_defaults:
        settings = { # dictionary to display for better analysis
                "Experiment info": get_new_exp_info_dict(),
                "Cognitive parameters": get_new_cog_dict(), #supply no values for default
                "Other parameters": get_other_params_dict(),
                "Environment": {},
                "Extra": {},
            }
        return settings
    if exp_info_dict==None: exp_info_dict=get_new_exp_info_dict()
    if cog_param_dict==None: cog_param_dict=get_new_cog_dict()
    if other_param_dict==None: other_param_dict=get_other_params_dict()
    settings = {  # dictionary to display for better analysis
        "Experiment info": exp_info_dict,
        "Cognitive parameters": cog_param_dict,
        "Other parameters": other_param_dict,
        "Environment": { #this is too environment specific to set here, will be set later

        },
        "Extra": { #added during runtime

        },
    }
    # for extra parts added during calculation, see bottom of this file
    return settings



def state_point_to_index(width, x, y):
    return y * width + x

def state_index_to_point(width, index_state):
    x = index_state % width
    y = index_state // width
    return x, y

def vertical_road_indices(start, stop, width, height):
    indices = [start]
    last_index = start
    while last_index < stop:
        next_index = last_index + width
        indices.append(next_index)
        last_index = indices[-1]
    return indices


def compute_roads(horizontal_list, vertical_list, width, height):
    length = 0
    horizontal_indices = []
    for h in horizontal_list: #always a start and end point
        all_indices = range(h[0], h[1])
        length += len(all_indices)
        horizontal_indices += all_indices

    vertical_indices = []
    for v in vertical_list:
        all_indices = vertical_road_indices(v[0], v[1], width, height)
        length+= len(all_indices)
        vertical_indices += all_indices

    return length, horizontal_indices, vertical_indices


#
# "Extra": {
#     "n_states": self.n_states,
#     "num road cells": len(self.road_indices),
#     "ratio of usable states": len(self.road_indices)/self.n_states,
#     "sum_r1": np.sum(self.rp_1),
#     "sum_r2": np.sum(r2),
#     "sum_positive_rewards": np.sum(self.r[self.r > 0]),
#     "sum_negative_rewards": np.sum(self.r[self.r < 0]),
#     "sum_r": np.sum(self.r),
#     "simple_rp_1": self.rp_1
# }
# ""
# "Results" = {
#             "cosine_sim_dict":  {
#                 "avg_sim_irl_exp": avg_sim_irl_exp,
#                 "avg_sim_irl_opt": avg_sim_irl_opt,
#                 "avg_sim_exp_opt": avg_sim_exp_opt,
#             },
#             "rewards_dict": rewards_dict = {
#                 "rewards_expert": np.round(rewards_expert, 2),
#                 "r1_expert": np.round(r1_expert, 2),
#                 "r2_expert": np.round(r2_expert, 2),
#                 "rewards_irl": np.round(rewards_irl, 2),
#                 "r1_irl": np.round(r1_irl, 2),
#                 "r2_irl": np.round(r2_irl, 2),
#                 "rewards_optimal": np.round(rewards_optimal, 2),
#                 "r1_optimal": np.round(r1_optimal, 2),
#                 "r2_optimal": np.round(r2_optimal, 2)
#             },
#             "expert_st_policy": expert_policy,
#             "expert_trajectories": expert_trajectory_states,
#             "agent_st_policy": agent_policy,
#             "agent_trajectories": agent_trajectory_states,
#             "value_it_irl": value_it_irl,
#             "optimal_st_policy": optimal_st_policy,
#             "optimal_det_policy": optimal_det_policy,
#             "optimal_trajectories": optimal_trajectory_states,
#             "reward_maxent": reward_maxent,
#             "e_svf": e_svf,
#             "e_features": e_features,
#             "trajectory_feature_expectation": e_features,
#             "maxent_feature_expectation": features.T.dot(e_svf),
#         }
#
# "Cognitive Calculations" = {
#             "cognitive_distortion" : self.cognitive_distortion,
#             "reward_arr1_o": self.reward_arr1_o,
#             "reward_arr2_o": self.reward_arr2_o,
#             "simple_p1": self.simple_p1,
#             "simple_rp_1": self.simple_rp_1,
#             "simple_r": self.simple_r,
#             "v2_o": self.v2_o,          # v2 is always deterministic here, assumption caused by experiment design
#             "v1_o": self.v1_o,
#             "simple_v": self.simple_v,
#             "subjective": {
#                 "r1_subj_r": self.r1_subj_r,
#                 "r1_subj_p": self.r1_subj_p,
#                 "r1_subj_all": self.r1_subj_all,
#                 "v1_subj_v": self.v1_subj_v,
#                 "v1_comb_subj_all": self.v1_comb_subj_all,
#                 "v2_comb_subj_all": self.v2_comb_subj_all,
#                 "value_it_1_and_2_soph_subj_all": self.value_it_1_and_2_soph_subj_all,
#             },
#             "objective": {
#                 "v1_comb_o": self.v1_comb_o,
#                 "v2_comb_o": self.v2_comb_o,
#                 "value_it_1_and_2_soph_o": self.value_it_1_and_2_soph_o,
#                 "v_comb_o_cccost": self.v_comb_o_cccost
#             }
#         }