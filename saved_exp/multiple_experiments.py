import itertools

import rp1.saved_exp.default_values as defaults
import rp1.saved_exp.exp_square_roads as square_road_exp
import rp1.saved_exp.exp_poc as exp_poc

import copy

which_experiment=0

chooseExpDict = {
    "0": "exp_poc",
    "1": "exp_square_roads",

}

def get_place_rewards(traffic_probability, punishment, prize, tiny_prize, very_tiny_prize): #i, j, p, t
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
        "reward values": {
            "punishment": punishment,
            "prize": prize,
            "tiny_prize": tiny_prize,
            "traffic_probability": traffic_probability,
            "very_tiny_prize": very_tiny_prize
        }
    }
    return places_and_rewards_dict


def execute_chosen_experiment():
    exp_name = chooseExpDict[str(which_experiment)]
    if exp_name == "exp_poc":
        execute_poc_exp_cog_params()
    elif exp_name == "exp_square_roads":
        experiment_place_rewards_square()


def execute_poc_exp_cog_params():
    alphas = [1.0, 0.6]
    betas = [1.0, 0.6]
    kappas = [4.0, 2.0]
    etas = [0.4, 0.8]
    cc_constants = [0.5, 0.7, 0.9, 1.2, 1.5, 2.5, 3.5, 5.0]
    baselines = [0.0]

    punishment = -10
    traffic_prob = 0.3
    prize = 10
    tiny_prize = 2
    very_tiny_prize = 1.0 #currently not in use

    #places rewards static for now
    new_places_rewards_dict=get_place_rewards(traffic_probability=traffic_prob, punishment=punishment, prize=prize, tiny_prize=tiny_prize, very_tiny_prize=very_tiny_prize)

    trial_no = 8000
    for item in itertools.product(alphas, betas, kappas, etas, cc_constants, baselines):
        print("trial_no", trial_no)
        print(item)
        #print(item)
        gamma1=0.8 #time discount 1
        gamma2=0.8
        baseline_changes=False
        values_tested = "cog params: " + " alpha " + str(item[0]) + \
                                                                 " beta " + str(item[1]) + " kappa " + str(item[2]) + \
                                                                 " eta " + str(item[3]) + " cc_constant " + str(item[4]) + \
                                                                 " baseline " + str(item[5]) + " time_disc_1 " + str(gamma1) + " time_disc_2 " + str(gamma2)
        exp_info_dict = defaults.get_new_exp_info_dict(exp_name="poc world with different cognitive params",
                                                       what=values_tested, mode="subjective", trial_no=trial_no)
        new_other_params = defaults.get_other_params_dict(policy_weighting= lambda x: x**30, number_of_expert_trajectories=100) #rest is default values

        new_cog_dict = defaults.get_new_cog_dict(alpha=item[0], beta=item[1], kappa=item[2], eta=item[3], time_disc_1=gamma1, time_disc_2=gamma2,
                        cc_constant=item[4], baseline=item[5], baseline_changes=baseline_changes)
        irl, start_settings_all = exp_poc.get_exp(populate_all_with_defaults=False, exp_info_dict=exp_info_dict, other_param_dict=new_other_params,
                                                         places_rewards_dict=new_places_rewards_dict, cognitive_update_dict=new_cog_dict,
                                                         visualize=True)
        irl.perform_irl()
        trial_no += 1


def experiment_place_rewards_square():
    traffic_probabilities = [0.8]
    punishments = [-1, -5, -10]
    prizes = [20]
    tiny_prizes = [1]
    trial_no = 6000
    very_tiny_prize = 0.1 #currently not in use
    cognitive_control_costs = [0.5, 0.7, 0.9, 1.2, 1.5, 1.8]
    for item in itertools.product(traffic_probabilities, punishments, prizes, tiny_prizes, cognitive_control_costs):
        k = item[0]
        j = item[1]
        p = item[2]
        t = item[3]
        cc = item[4]
        places_and_rewards_dict = get_place_rewards(k, j, p, t, very_tiny_prize)
        values_tested=" punishment = " + str(j) + " prize = " + str(p) + \
                      " tiny_prize = " + str(t) + " traffic_probability = " + str(k) + \
                      " cc_cost = " + str(cc)
        exp_info_dict = defaults.get_new_exp_info_dict(exp_name="square roads with different rewards and probs",
                                                       what=values_tested, mode="subjective", trial_no=trial_no)
        trial_no+=1
        # using default other params, including "eliminate loops in trajectory": eliminate_loops=True
        # using default cognitive model, just altering cc constant
        new_cogn = defaults.get_new_cog_dict(cc_constant=cc)
        irl, start_settings_all = square_road_exp.get_exp(populate_all_with_defaults=False,
                                                          cognitive_update_dict=new_cogn, exp_info_dict=exp_info_dict,
                                                          places_rewards_dict=places_and_rewards_dict,
                                                          visualize=True)
        print(start_settings_all)

        irl.perform_irl()


execute_chosen_experiment()


