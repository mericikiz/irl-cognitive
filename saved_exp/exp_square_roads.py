import rp1.saved_exp.default_values as defaults
from rp1.gridenv import GridEnvironment
import numpy as np
from rp1.cognitive import Cognitive_model
from rp1.irl import IRL_cognitive


#______________DEFINE THESE + ANYTHING YOU WANT TO OVERRIDE

start = [0]
semi_target = [54]
terminal = [99]
width = 10
height = 10
vertical_roads = [[0, 90], [4, 94], [39, 99]]
horizontal_roads = [[0, 4], [90, 99], [34, 39]]
deterministic = False

road_length, horizontal_indices, vertical_indices = defaults.compute_roads(horizontal_roads, vertical_roads, width, height)


def get_default_places_rewards():
    print("place reward settings is None, using default values")
    traffic_probability=0.5
    punishment = -5
    prize = 20
    tiny_prize = 5
    very_tiny_prize = 1
    this_exp_dict_default = {
        "traffic": {
            "p": traffic_probability,  # probability of r1
            "r1": punishment,
            "r2": 0  # we can put a baseline value later as well
        },
        "home": {
            "r1": tiny_prize,
            "p": 1.0,  # probability of r1
            "r2": tiny_prize
        },
        "bank": {
            "r1": tiny_prize,
            "p": 1.0,  # probability of r1
            "r2": prize
        },
        "agent": {  # a special  case
            "r1": 0,
            "r2": 0  # starting reward state
        },
        "reward values": {
            "punishment": punishment,
            "prize": prize,
            "tiny_prize": tiny_prize,
            "traffic_probability": traffic_probability,
            "very_tiny_prize": very_tiny_prize
        }
    }
    return this_exp_dict_default

def get_default_env_settings(places_and_rewards):
    exp_dict = {
            "deterministic" : deterministic,
            "width" : width,
            "height" : height,
            "traffic_probability": places_and_rewards["reward values"]["traffic_probability"],
            "punishment" : places_and_rewards["reward values"]["punishment"],
            "prize" : places_and_rewards["reward values"]["prize"],
            "tiny_prize" : places_and_rewards["reward values"]["tiny_prize"],
            "very_tiny_prize" : places_and_rewards["reward values"]["very_tiny_prize"],
            "start": start,
            "terminal": terminal,
            "semi target": semi_target,
            "places_list" : ["home", "bank", "agent"],  # traffic is counted as part of roads, not place
            "start_states": start,
            "terminal_states": terminal,  # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
            "traffic": {
                "indices": defaults.vertical_road_indices(14, 74, width, height)+[35, 36],
                "p": places_and_rewards["traffic"]["p"],  # probability of r1
                "r1": places_and_rewards["traffic"]["r1"],
                "r2": 0 #we can put a baseline value later as well
            },
            "home": {
                "indices": [99],
                "r1": places_and_rewards["home"]["r1"],
                "p": places_and_rewards["home"]["p"],  # probability of r1
                "r2": places_and_rewards["home"]["r2"]
            },
            "bank": {
                "indices": [54],
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
                "vertical": vertical_indices,
                "horizontal": horizontal_indices,
                "road_length": road_length
            },

        }
    return exp_dict



def make_r1(this_exp_dict, p=None):
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


def make_r2(this_exp_dict):
    reward_array2 = np.zeros(height*width)
    reward_array2[this_exp_dict["traffic"]["indices"]] = this_exp_dict["traffic"]["r2"]
    reward_array2[this_exp_dict["home"]["indices"]] = this_exp_dict["home"]["r2"]
    reward_array2[this_exp_dict["bank"]["indices"]] = this_exp_dict["bank"]["r2"]
    return reward_array2


def get_exp(populate_all_with_defaults=False, exp_info_dict=None, other_param_dict=None, places_rewards_dict=None, cognitive_update_dict=None, visualize=False):
    if populate_all_with_defaults:
        start_settings_all=defaults.get_settings(populate_with_defaults=True)
    else:
        start_settings_all=defaults.get_settings(populate_with_defaults=False, exp_info_dict=exp_info_dict,
                                                 cog_param_dict=cognitive_update_dict, other_param_dict=other_param_dict)
    if places_rewards_dict == None:
        places_rewards_dict=get_default_places_rewards()
    env_settings_all = get_default_env_settings(places_rewards_dict)
    start_settings_all["Environment"] = env_settings_all
    print("debug square roads")
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




