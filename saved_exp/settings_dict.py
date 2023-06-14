import numpy as np


def make_experiment(start, terminal,
                    traffic_indices, traffic_p, traffic_r,):
    visual_dict = {
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







    settings = { # dictionary to display for better analysis
        "Experiment info": {
            #"exp setup no": exp_no,
            #"description env": description,
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
            "semi target": semi_target,
            "mode" : mode
        },
    }