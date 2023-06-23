from rp1.saved_exp.default_values import *
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

road_length, horizontal_indices, vertical_indices = compute_roads(horizontal_roads, vertical_roads, width, height)

punishment = -5
prize = 20
tiny_prize = 5
very_tiny_prize = 1

def set_exp_settings(places_and_rewards, punishment=-5.0, prize=20.0, tiny_prize=5.0, traffic_probability= 0.95, very_tiny_prize=1.0):
    exp_dict = {
            "deterministic" : deterministic,
            "width" : width,
            "height" : height,
            "punishment" : punishment,
            "prize" : prize,
            "tiny_prize" : tiny_prize,
            "very_tiny_prize" : very_tiny_prize,
            "start": start,
            "terminal": terminal,
            "semi target": semi_target,
            "places_list" : ["home", "bank", "agent"],  # traffic is counted as part of roads, not place
            "start_states": start,
            "terminal_states": terminal,  # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
            "traffic": {
                "indices": vertical_road_indices(14, 74, width, height)+[35, 36],
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

def get_exp(settings_exp, this_exp_dict, visualize=False):

    settings_exp["Environment"] = this_exp_dict
    settings_exp["Experiment info"]["exp name"] = "Square world with 3 main roads"
    env = GridEnvironment(settings_exp)
    r1, rp1 = make_r1(this_exp_dict)
    extra_info_dict = env.set_objective_r1_and_r2(r1, make_r2(this_exp_dict), rp1)
    settings_exp["Extra"] = extra_info_dict

    cognitive_model = Cognitive_model(env, alpha, beta, kappa, eta, time_disc_1, time_disc_2, cc_constant, baseline, subjective)

    irl = IRL_cognitive(env, cognitive_model, settings_exp, visualize=visualize)
    return irl



