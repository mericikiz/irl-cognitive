from rp1.gridenv import GridEnvironment
import numpy as np
from rp1.cognitive import Cognitive_model
from rp1.irl import IRL_cognitive


visualize = True

#_______________DESCRIPTION________________
exp_name = "loss aversion, low cognitive control cost, high sensitivity to losses" #not displayed because too long
exp_no = 1
env_type = "Grid World"
description = "simple deterministic grid world"
RL_algorithm = "Value Iteration"


#______________COGNITIVE MODEL PARAMS_____________
#TODO add function type of cc constant, rn it is hard coded inside cognitive model as a linear function
time_disc_1 = 0.95
time_disc_2 = 0.95
alpha = 1.0 #α > 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 1.7 #β < 1: Indicates a concave value function, suggesting diminishing sensitivity to losses.
kappa = 3.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 1.5 #η > 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 2.0

#_____________OTHER HYPERPARAMETERS_____________
policy_weighting = lambda x: x**50 #lambda x: x #lambda x: x**50
number_of_expert_trajectories = 200
eliminate_loops = True

#_______________ENVIRONMENT AND REWARDS_____________
width = 5
height = 5
deterministic = True

start = [10]
terminal = [4, 24]

punishment = -7
prize = 20
tiny_prize = 5
very_tiny_prize = 1

#______________________IMAGES__________________________


#___________________TESTS DICTIONARY___________________
tests_dict = {
    "test subjective valuation": False,
    "test subjective probability": False,
    "test normalization": False,
}

#__________FOR ENVIRONMENT VISUALIZATION_____________
visual_dict = {
    "places_list" : ["traffic", "home", "bank", "agent"],
    "start_states": start,
    "terminal_states": terminal, # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
    "traffic": {
        "indices": [2, 3, 4, 7, 8, 9, 24],
        "r1": punishment,
        "r2": 0 #we can put a baseline value later as well
    },
    "home": {
        "indices": [24],
        "r1": tiny_prize,
        "r2": very_tiny_prize
    },
    "bank": {
        "indices": [4],
        "r1": tiny_prize,
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

def make_r1():
    reward_array1 = np.zeros(env.n_states)
    reward_array1[visual_dict["traffic"]["indices"]] = visual_dict["traffic"]["r1"]
    reward_array1[visual_dict["home"]["indices"]] = visual_dict["home"]["r1"]
    reward_array1[visual_dict["bank"]["indices"]] = visual_dict["bank"]["r1"]
    return reward_array1

def make_r2():
    reward_array2 = np.zeros(env.n_states)
    reward_array2[visual_dict["traffic"]["indices"]] = visual_dict["traffic"]["r2"]
    reward_array2[visual_dict["home"]["indices"]] = visual_dict["home"]["r2"]
    reward_array2[visual_dict["bank"]["indices"]] = visual_dict["bank"]["r2"]
    return reward_array2

settings = { # dictionary to display for better analysis
    "Experiment info": {
        "exp setup no": exp_no,
        "description env": description,
        "RL algorithm used": RL_algorithm
    },
    "Cognitive parameters": {
        "cognitive_control": cc_constant,
        "time_disc_1": time_disc_1,
        "time_disc_2": time_disc_2,
        "alpha" : alpha,
        "beta" : beta,
        "kappa" : kappa,
        "eta" : eta,
    },
    "Other parameters": {
        #"softmax temperature": softmax_temperature,
        "policy weighting" : policy_weighting,  # down-weighting of less optimal actions,
        "number of expert trajectories": number_of_expert_trajectories,
        "eliminate loops in trajectory": eliminate_loops
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

env = GridEnvironment(exp_name, width, height, deterministic, tests_dict, visual_dict)

env.set_objective_r1_and_r2(make_r1(), make_r2())

cognitive_model = Cognitive_model(env, alpha, beta, kappa, eta, time_disc_1, time_disc_2, cc_constant)

irl = IRL_cognitive(env, cognitive_model, settings)

irl.perform_irl(visualize=visualize)
