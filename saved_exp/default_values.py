import numpy as np
visualize = True

exp_name = ""
RL_algorithm = "Value Iteration"
what = "" #what is being tested

#______________COGNITIVE MODEL PARAMS_____________
time_disc_1 = 0.8
time_disc_2 = 0.8
alpha = 0.7 #α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 0.55 #β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses.
kappa = 2.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 0.8 #η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 2.0

#_____________OTHER HYPERPARAMETERS_____________
policy_weighting = lambda x: x**50 #lambda x: x #lambda x: x**50
number_of_expert_trajectories = 200
eliminate_loops = True

#_______________ENVIRONMENT AND REWARDS_____________
deterministic = False

mode = "subjective"
if mode == "subjective":
    subjective=True
else: subjective=False

def print_text_env(width, height):
    print("gridworld indices in text")
    np.set_printoptions(linewidth=80, edgeitems=width, suppress=True)
    arr = np.arange(0, width*height).reshape((height, width))

    print(arr[::-1]) #reversed order

def get_settings(width, height, punishment, prize, tiny_prize, very_tiny_prize, start, terminal, semi_target):
    settings = { # dictionary to display for better analysis
            "Experiment info": {
                "exp name": exp_name,
                "RL algorithm used": RL_algorithm,
                "what is being tested": what
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
                "policy weighting" : policy_weighting,  # down-weighting of less optimal actions,
                "number of expert trajectories": number_of_expert_trajectories,
                "eliminate loops in trajectory": eliminate_loops
            },
            "Environment": {
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
                "semi target": semi_target,
                "mode" : mode
            },
        }
    return settings