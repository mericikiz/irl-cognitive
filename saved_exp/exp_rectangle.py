# from rp1.gridenv import GridEnvironment
# import numpy as np
# from rp1.cognitive import Cognitive_model
# from rp1.irl import IRL_cognitive
#
#
# visualize = True
#
# exp_name = ""
# RL_algorithm = "Value Iteration"
# what = "" #what is being tested
#
#
#
# #______________COGNITIVE MODEL PARAMS_____________
# #TODO add function type of cc constant, rn it is hard coded inside cognitive model as a linear function
# time_disc_1 = 0.8
# time_disc_2 = 0.8
# alpha = 0.7 #α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
# beta = 0.55 #β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses.
# kappa = 2.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
# eta = 0.8 #η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
# cc_constant = 2.0
#
# #_____________OTHER HYPERPARAMETERS_____________
# policy_weighting = lambda x: x**50 #lambda x: x #lambda x: x**50
# number_of_expert_trajectories = 200
# eliminate_loops = True
#
# #_______________ENVIRONMENT AND REWARDS_____________
# width = 10
# height = 5
# deterministic = False
#
# start = [20]
# terminal = [49]
# semi_target = [15]
# mode = "subjective"
# if mode == "subjective":
#     subjective=True
# else: subjective=False
#
# punishment = -7
# prize = 20
# tiny_prize = 5
# very_tiny_prize = 1
#
# #______________________IMAGES__________________________
#
#
# settings = { # dictionary to display for better analysis
#         "Experiment info": {
#             "exp name": exp_name,
#             # "description env": description,
#             "RL algorithm used": RL_algorithm,
#             "what is being tested": what
#         },
#         "Cognitive parameters": {
#             "cognitive_control": cc_constant,
#             "time_disc_1": time_disc_1,
#             "time_disc_2": time_disc_2,
#             "alpha" : alpha,
#             "beta" : beta,
#             "kappa" : kappa,
#             "eta" : eta,
#         },
#         "Other parameters": {
#             #"softmax temperature": softmax_temperature,
#             "policy weighting" : policy_weighting,  # down-weighting of less optimal actions,
#             "number of expert trajectories": number_of_expert_trajectories,
#             "eliminate loops in trajectory": eliminate_loops
#         },
#         "Environment": {
#             "deterministic" : deterministic,
#             "width" : width,
#             "height" : height,
#             "punishment" : punishment,
#             "prize" : prize,
#             "tiny_prize" : tiny_prize,
#             "very_tiny_prize" : very_tiny_prize
#         },
#         "IRL": {
#             "start" : start,
#             "terminal" : terminal,
#             "semi target": semi_target,
#             "mode" : mode
#         },
#     }
#
# visual_dict = {
#         "places_list" : ["traffic", "home", "bank", "agent"],
#         "start_states": start,
#         "terminal_states": terminal, # we assume there is always some goal in terminal state anyway, unless the environment is limited by time instead
#         "traffic": {
#             "indices": [14, 24, 25, 26, 16, 4],
#             "p": 0.95,  # probability of r1
#             "r1": punishment,
#             "r2": 0 #we can put a baseline value later as well
#         },
#         "home": {
#             "indices": [49],
#             "r1": tiny_prize,
#             "p": 1.0,  # probability of r1
#             "r2": very_tiny_prize
#         },
#         "bank": {
#             "indices": [15],
#             "r1": tiny_prize,
#             "p": 1.0,  # probability of r1
#             "r2": prize
#         },
#         "agent" : { # a special  case
#             "indices": start, #although there is only one at a time in current setup, it is in a list to fit with the format,
#             "r1": 0,
#             "r2": 0 #starting reward state
#         }
#         #"obstacle": { # cant walk through obstacles
#         #    "indices": [],
#         #},
#         #"agent": {} we can have a section describing agent's situation
#     }
#
# # testing effect of traffic probability
# env = GridEnvironment(exp_name, width, height, deterministic, visual_dict)
# to_test = [0.05, 0.2, 0.5, 0.7, 0.95]
# for i in to_test:
#     r1, rp1 = make_r1(i)
#     env.set_objective_r1_and_r2(r1, make_r2(), rp1)
#     cognitive_model = Cognitive_model(env, alpha, beta, kappa, eta, time_disc_1, time_disc_2, cc_constant, subjective)
#     irl = IRL_cognitive(env, cognitive_model, settings)
#     print("mode:", mode)
#
#     irl.perform_irl(visualize=visualize)
