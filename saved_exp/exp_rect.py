from rp1.saved_exp.default_values import *
from rp1.gridenv import GridEnvironment
import numpy as np
from rp1.cognitive import Cognitive_model
from rp1.irl import IRL_cognitive
import json


#______________DEFINE THESE + ANYTHING YOU WANT TO OVERRIDE
start = [20]
semi_target = [15]
terminal = [49]

punishment = -7
prize = 20
tiny_prize = 5
very_tiny_prize = 1
width = 10
height = 5

exp_dict = {
        "places_list" : ["traffic", "home", "bank", "agent"],
        "start_states": start,
        "terminal_states": terminal, # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
        "traffic": {
            "indices": [14, 24, 25, 26, 16, 4],
            "p": 0.95,  # probability of r1
            "r1": punishment,
            "r2": 0 #we can put a baseline value later as well
        },
        "home": {
            "indices": [49],
            "r1": tiny_prize,
            "p": 1.0,  # probability of r1
            "r2": very_tiny_prize
        },
        "bank": {
            "indices": [15],
            "r1": tiny_prize,
            "p": 1.0,  # probability of r1
            "r2": prize
        },
        "agent" : { # a special  case
            "indices": start, #although there is only one at a time in current setup, it is in a list to fit with the format,
            "r1": 0,
            "r2": 0 #starting reward state
        }
        #"obstacle": { # cant walk through obstacles
        #    "indices": [],
        #},
        #"agent": {} we can have a section describing agent's situation
    }

def make_r1(p=None):
    if deterministic:
        p1 = np.ones(width * height)
    else:
        p1 = np.ones(width * height)
        if p == None: p = exp_dict["traffic"]["p"]
        p1[exp_dict["traffic"]["indices"]] = p
        p1[exp_dict["home"]["indices"]] = exp_dict["home"]["p"]
        p1[exp_dict["bank"]["indices"]] = exp_dict["bank"]["p"]
    reward_array1 = np.zeros(width*height)
    reward_array1[exp_dict["traffic"]["indices"]] = exp_dict["traffic"]["r1"]
    reward_array1[exp_dict["home"]["indices"]] = exp_dict["home"]["r1"]
    reward_array1[exp_dict["bank"]["indices"]] = exp_dict["bank"]["r1"]
    return reward_array1, p1


def make_r2():
    reward_array2 = np.zeros(height*width)
    reward_array2[exp_dict["traffic"]["indices"]] = exp_dict["traffic"]["r2"]
    reward_array2[exp_dict["home"]["indices"]] = exp_dict["home"]["r2"]
    reward_array2[exp_dict["bank"]["indices"]] = exp_dict["bank"]["r2"]
    return reward_array2

def get_exp():
    settings = get_settings(width, height, punishment, prize, tiny_prize, very_tiny_prize, start, terminal, semi_target)
    env = GridEnvironment(exp_name, width, height, deterministic, exp_dict)
    r1, rp1 = make_r1()
    env.set_objective_r1_and_r2(r1, make_r2(), rp1)

    cognitive_model = Cognitive_model(env, alpha, beta, kappa, eta, time_disc_1, time_disc_2, cc_constant, subjective)

    irl = IRL_cognitive(env, cognitive_model, settings)
    print("mode:", mode)

    irl.perform_irl(visualize=visualize)


