from rp1.gridenv import GridEnvironment
import numpy as np
from rp1.cognitive import Cognitive_model
from rp1.irl import IRL_cognitive
from rp1.visualizations_all import Visuals

#_______________DESCRIPTION________________
exp_name = "loss aversion, low cognitive control cost, high sensitivity to losses"
env_type = "Grid World"
description = "simple deterministic grid world"
immediate_visualize = True
RL_algorithm = "Value Iteration"


#______________COGNITIVE MODEL PARAMS_____________
#TODO add function type of cc constant, rn it is hard coded inside cognitive model as a linear function
time_disc_1 = 0.9
time_disc_2 = 0.7
alpha = 1.0 #α > 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 1.7 #β < 1: Indicates a concave value function, suggesting diminishing sensitivity to losses.
kappa = 3.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 1.5 #η > 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 0.6


#_______________ENVIRONMENT AND REWARDS_____________
width = 5
height = 5
deterministic = True

start = [0]
terminal = [24]

punishment = -9
prize = 10
tiny_prize = 1
very_tiny_prize = 0.1

#______________________IMAGES__________________________
img_home = "images/home.png"
img_donut = "images/donut.png"


#___________________TESTS DICTIONARY___________________
tests_dict = {
    "test_subjective_valuation": True,
    "test_subjective_probability": False,
    "test_normalization": False,
}


def make_r1():
    reward_array1 = np.zeros(env.n_states)
    reward_array1[3] = punishment
    reward_array1[2] = punishment
    reward_array1[8] = punishment
    reward_array1[7] = punishment
    reward_array1[9] = punishment
    reward_array1[4] = tiny_prize
    reward_array1[24] = tiny_prize
    return reward_array1

def make_r2():
    reward_array2 = np.zeros(env.n_states)
    reward_array2[4] = prize
    reward_array2[3] = tiny_prize
    reward_array2[2] = tiny_prize
    reward_array2[8] = tiny_prize
    reward_array2[7] = tiny_prize
    reward_array2[9] = tiny_prize
    reward_array2[24] = very_tiny_prize
    return reward_array2

env = GridEnvironment(exp_name, width, height, deterministic, tests_dict)

env.set_objective_r1_and_r2(make_r1(), make_r2())

cognitive_model = Cognitive_model(env, alpha, beta, kappa, eta, time_disc_1, time_disc_2, cc_constant)

settings = { # dictionary to display for better analysis
    "Experiment info": {
        "name": exp_name,
        "description": description,
        "RL_algorithm used": RL_algorithm
    },
    "Cognitive parameters": {
        "cc_constant": cc_constant,
        "time_disc_1": time_disc_1,
        "time_disc_2": time_disc_2,
        "alpha" : alpha,
        "beta" : beta,
        "kappa" : kappa,
        "eta" : eta,
    },
    "Environment": {
        "type": env_type,
        "deterministic" : deterministic,
        "width" : width,
        "height" : height,
        "punishment" : punishment,
        "prize" : prize,
        "tiny_prize" : tiny_prize,
        "very_tiny_prize" : very_tiny_prize
    },
    "IRL": {
        "start" : start,
        "terminal" : terminal,
    },
    "Tests" : tests_dict
}

irl = IRL_cognitive(env, cognitive_model, settings)

vis = Visuals(irl, save_bool=True, show=False)
if immediate_visualize:
    vis.visualize_initials()




