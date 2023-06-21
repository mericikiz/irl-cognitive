import rp1.saved_exp.default_values as default_settings
import rp1.saved_exp.exp_square_roads as square_road_exp
import copy


traffic_probability = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99 ]
traffic_punishment = [-0.5, -1.0, -2.5, -5.0, -7.5, -9, -9.5]
prizes_bank = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
tiny_prize = [1.0, 5.0, 10.0]


for i in traffic_probability:
    for j in traffic_punishment:
        for p in prizes_bank:
            for t in tiny_prize:
                this_exp_dict = copy.deepcopy(square_road_exp.set_exp_settings(punishment=j, prize=p, tiny_prize=t, traffic_probability=i))
                set_general = copy.deepcopy(default_settings.get_settings())
                set_general["Experiment info"]["what is being tested"] = " punishment = " + str(j) + " prize = " + str(p) + " tiny_prize = " + str(t) + " traffic_probability " + str(i)
                irl = square_road_exp.get_exp(set_general, this_exp_dict)
                irl.perform_irl()
