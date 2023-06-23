import numpy as np



exp_name = ""
RL_algorithm = "Value Iteration"
what = "" #what is being tested

#______________COGNITIVE MODEL PARAMS_____________
time_disc_1 = 0.9
time_disc_2 = 0.9
alpha = 0.7 #α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 0.55 #β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses.
kappa = 2.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 0.8 #η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 2.0
baseline = 0.0

#_____________OTHER HYPERPARAMETERS_____________
policy_weighting = lambda x: x**20 # how much more likely to take more optimal actions at a given time
number_of_expert_trajectories = 50
eliminate_loops = True

mode = "subjective"
if mode == "subjective":
    subjective=True
else: subjective=False

def print_text_env(width, height):
    print("gridworld indices in text")
    np.set_printoptions(linewidth=80, edgeitems=width, suppress=True)
    arr = np.arange(0, width*height).reshape((height, width))

    print(arr[::-1]) #reversed order

def get_settings():
    settings = { # dictionary to display for better analysis
            "Experiment info": {
                "exp name": exp_name,
                "RL algorithm used": RL_algorithm,
                "what is being tested": what,
                "mode": mode
            },
            "Cognitive parameters": {
                "cognitive_control": cc_constant,
                "time_disc_1": time_disc_1,
                "time_disc_2": time_disc_2,
                "alpha" : alpha,
                "beta" : beta,
                "kappa" : kappa,
                "eta" : eta,
                "baseline": baseline
            },
            "Other parameters": {
                "policy weighting" : policy_weighting,  # down-weighting of less optimal actions,
                "number of expert trajectories": number_of_expert_trajectories,
                "eliminate loops in trajectory": eliminate_loops
            },
            "Environment": {

            },
            "Extra": {

            },
        }
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