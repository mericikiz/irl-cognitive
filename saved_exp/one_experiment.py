import rp1.saved_exp.default_values as default_settings
import rp1.saved_exp.exp_square_roads as square_road_exp
import copy
#______________COGNITIVE MODEL PARAMS_____________
time_disc_1 = 0.9
time_disc_2 = 0.9
alpha = 0.7 #α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 0.55 #β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses.
kappa = 1.5 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 0.8 #η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 1.0
baseline = 0.0
cog_param_dict = {
    "cognitive_control": cc_constant,
    "time_disc_1": time_disc_1,
    "time_disc_2": time_disc_2,
    "alpha": alpha,
    "beta": beta,
    "kappa": kappa,
    "eta": eta,
    "baseline": baseline
}

punishment = -1
prize = 30 #prize
tiny_prize = 1 #tiny_prize
traffic_probability =0.5

places_and_rewards_dict = {
    "traffic": {
        "p": traffic_probability,  # probability of r1
        "r1": punishment,
        "r2": 0 #we can put a baseline value later as well
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
    "agent" : { # a special  case
        "r1": 0,
        "r2": 0 #starting reward state
    },

}

this_exp_dict = copy.deepcopy(square_road_exp.set_exp_settings(places_and_rewards_dict, punishment=punishment, prize=prize, tiny_prize=tiny_prize, traffic_probability=traffic_probability))
set_general = copy.deepcopy(default_settings.get_settings())
set_general["Experiment info"]["what is being tested"] = " punishment = " + str(punishment) + " prize = " + str(prize) + " tiny_prize = " \
                                                         + str(tiny_prize) + " traffic_probability " + str(traffic_probability)
print(set_general["Experiment info"]["what is being tested"])
set_general["Cognitive parameters"] = cog_param_dict
irl = square_road_exp.get_exp(set_general, this_exp_dict, visualize=True)
irl.perform_irl()