import rp1.saved_exp.default_values as default_settings
import rp1.saved_exp.exp_square_roads as square_road_exp
import copy


traffic_probability_list = [0.3, 0.05, 0.5, 0.7, 0.95]
prize_list = [20, 30]
tiny_prize_list = [1, 5]
punishment_list = [-3] #[-1, -5, -10]


trial_no=61

def get_place_rewards(traffic_probability, punishment, prize, tiny_prize): #i, j, p, t
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
    return places_and_rewards_dict


for i in traffic_probability_list:
    for j in punishment_list:
        for p in prize_list:
            for t in tiny_prize_list:
                places_and_rewards_dict = get_place_rewards(i, j, p, t)
                this_exp_dict = copy.deepcopy(square_road_exp.set_exp_settings(places_and_rewards=places_and_rewards_dict,
                                                                               punishment=j, prize=p, tiny_prize=t, traffic_probability=i))
                set_general = copy.deepcopy(default_settings.get_settings())
                set_general["Experiment info"]["what is being tested"] = " punishment = " + str(j) + " prize = " + str(p) + " tiny_prize = " + str(t) + " traffic_probability " + str(i)
                set_general["Experiment info"]["trial no"] = trial_no
                print("trial_no", trial_no)
                print(set_general["Experiment info"]["what is being tested"])
                trial_no += 1
                irl = square_road_exp.get_exp(set_general, this_exp_dict, visualize=True)
                irl.perform_irl()