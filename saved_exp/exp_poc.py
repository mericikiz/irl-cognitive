import rp1.saved_exp.default_values as defaults
from rp1.gridenv import GridEnvironment
import numpy as np
from rp1.cognitive import Cognitive_model
from rp1.irl import IRL_cognitive
import copy

#______________DEFINE THESE + ANYTHING YOU WANT TO OVERRIDE

start = [10]
semi_target = [4]
terminal = [24, 4]
width = 5
height = 5
vertical_roads = []
horizontal_roads = []
deterministic = False

# time_disc_1 = 0.8
# time_disc_2 = 0.8
# alpha = 0.8 #α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
# beta = 0.8 #β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses.
# kappa = 1.5 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
# eta = 0.9 #η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
# cc_constant = 2.0
# baseline = 0.0


def get_indv_places_rewards(traffic_probability=0.5, punishment=-5, prize=20, tiny_prize=5, very_tiny_prize=1): #i, j, p, t
    places_and_rewards_dict = {
        "traffic": {
            "p": traffic_probability,  # probability of r1
            "r1": punishment,
            "r2": 0 #we can put a baseline value later as well
        },
        "home": {
            "r1": prize,
            "p": 1.0,  # probability of r1
            "r2": very_tiny_prize
        },
        "bank": {
            "r1": tiny_prize,
            "p": 1.0,  # probability of r1
            "r2": prize
        },
        "agent" : { # a special  case
            "r1": 0,
            "r2": 0 #starting reward state
        },
        "reward values": {
            "punishment": punishment,
            "prize": prize,
            "tiny_prize": tiny_prize,
            "traffic_probability": traffic_probability,
            "very_tiny_prize": very_tiny_prize
        }
    }
    return places_and_rewards_dict


def get_default_env_settings(places_and_rewards):
    exp_dict = {
            "deterministic" : deterministic,
            "width" : width,
            "height" : height,
            "traffic_probability": places_and_rewards["reward values"]["traffic_probability"],
            "punishment": places_and_rewards["reward values"]["punishment"],
            "prize": places_and_rewards["reward values"]["prize"],
            "tiny_prize": places_and_rewards["reward values"]["tiny_prize"],
            "very_tiny_prize": places_and_rewards["reward values"]["very_tiny_prize"],
            "start": start,
            "terminal": terminal,
            "semi target": semi_target,
            "places_list" : ["home", "bank", "agent"],  # traffic is counted as part of roads, not place
            "start_states": start,
            "terminal_states": terminal,  # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
            "traffic": {
                "indices": [3, 8, 12, 13, 14],
                "p": places_and_rewards["traffic"]["p"],  # probability of r1
                "r1": places_and_rewards["traffic"]["r1"],
                "r2": 0 #we can put a baseline value later as well
            },
            "home": {
                "indices": [24],
                "r1": places_and_rewards["home"]["r1"],
                "p": places_and_rewards["home"]["p"],  # probability of r1
                "r2": places_and_rewards["home"]["r2"]
            },
            "bank": {
                "indices": [4],
                "r1": places_and_rewards["bank"]["r1"],
                "p": places_and_rewards["bank"]["p"],  # probability of r1
                "r2": places_and_rewards["bank"]["r2"]
            },
            "agent" : { # a special  case
                "indices": start, #although there is only one at a time in current setup, it is in a list to fit with the format,
                "r1": places_and_rewards["agent"]["r1"],
                "r2": places_and_rewards["agent"]["r2"] #starting state reward
            },
            "roads": {
                "vertical": vertical_roads,
                "horizontal": horizontal_roads,
                "road_length": 0
            },

        }
    return exp_dict


def make_r1(this_exp_dict=None, p=None):
    if deterministic:
        p1 = np.ones(width * height)
    else:
        p1 = np.ones(width * height)
        if p == None: p = this_exp_dict["traffic"]["p"]
        p1[this_exp_dict["traffic"]["indices"]] = p
        p1[this_exp_dict["home"]["indices"]] = this_exp_dict["home"]["p"]
        p1[this_exp_dict["bank"]["indices"]] = this_exp_dict["bank"]["p"]
    reward_array1 = np.zeros(width*height)
    reward_array1[this_exp_dict["traffic"]["indices"]] = this_exp_dict["traffic"]["r1"]
    reward_array1[this_exp_dict["home"]["indices"]] = this_exp_dict["home"]["r1"]
    reward_array1[this_exp_dict["bank"]["indices"]] = this_exp_dict["bank"]["r1"]
    return reward_array1, p1


def make_r2(this_exp_dict=None):
    reward_array2 = np.zeros(height*width)
    reward_array2[this_exp_dict["traffic"]["indices"]] = this_exp_dict["traffic"]["r2"]
    reward_array2[this_exp_dict["home"]["indices"]] = this_exp_dict["home"]["r2"]
    reward_array2[this_exp_dict["bank"]["indices"]] = this_exp_dict["bank"]["r2"]
    return reward_array2

def get_exp(populate_all_with_defaults=False, exp_info_dict=None, other_param_dict=None, places_rewards_dict=None, cognitive_update_dict=None, visualize=False):
    if populate_all_with_defaults:
        start_settings_all=defaults.get_settings(populate_with_defaults=populate_all_with_defaults)
    else:
        start_settings_all=defaults.get_settings(populate_with_defaults=False, exp_info_dict=exp_info_dict,
                                                 cog_param_dict=cognitive_update_dict, other_param_dict=other_param_dict)
    if places_rewards_dict == None:
        places_rewards_dict=get_indv_places_rewards()
    env_settings_all = get_default_env_settings(places_rewards_dict)
    start_settings_all["Environment"] = env_settings_all
    print("debug")
    print(start_settings_all)
    env = GridEnvironment(start_settings_all)
    r1, rp1 = make_r1(env_settings_all)
    extra_info_dict = env.set_objective_r1_and_r2(r1, make_r2(env_settings_all), rp1)
    start_settings_all["Extra"] = extra_info_dict
    if start_settings_all["Experiment info"]["mode"] == "subjective":
        subj_bool = True
    else:
        subj_bool = False

    cognitive_model = Cognitive_model(env, start_settings_all["Cognitive parameters"]["alpha"], start_settings_all["Cognitive parameters"]["beta"],
                                      start_settings_all["Cognitive parameters"]["kappa"], start_settings_all["Cognitive parameters"]["eta"],
                                      start_settings_all["Cognitive parameters"]["time_disc_1"], start_settings_all["Cognitive parameters"]["time_disc_2"],
                                      start_settings_all["Cognitive parameters"]["cognitive_control"], start_settings_all["Cognitive parameters"]["baseline"],
                                      subj_bool)
    irl = IRL_cognitive(env, cognitive_model, start_settings_all, visualize=visualize)
    return irl, start_settings_all



def perform_one_exp():
    trial_no=6666



    my_exp_info_dict = defaults.get_new_exp_info_dict(exp_name="POC with cognitive", what="poc visual",
                                                      trial_no=trial_no,
                                                      RL_algorithm="Value Iteration", mode="subjective")

    cog_dict = defaults.get_new_cog_dict(alpha=1.0, beta=1.0, kappa=4.0, eta=0.4, time_disc_1=0.7, time_disc_2=0.7, cc_constant=3.2)
    places_r_list = get_indv_places_rewards(traffic_probability=0.3, punishment=-10, prize=10, tiny_prize=4, very_tiny_prize=2)
    other_params = defaults.get_other_params_dict(number_of_expert_trajectories=200, policy_weighting=lambda x:x**40)


    irl, start_settings = get_exp(exp_info_dict=my_exp_info_dict, other_param_dict=other_params,
                                  places_rewards_dict=places_r_list, cognitive_update_dict=cog_dict, visualize=True)
    print(start_settings)
    trial_no += 1
    irl.perform_irl(save_intermediate_guessed_rewards=True)

perform_one_exp()