import json
import os
import rp1.evaluation_utils.exp_result_pointers as exps
import plotly.graph_objects as px
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import time

'''
exp1_bigchunk:
    - the test is different values/combintions of punishment, prize, tiny_prize, traffic_probability
    
'''
impossible_states = [5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 28, 29, 31, 32, 33, 41, 42, 43, 45, 46, 47, 48, 51, 52, 53, 55, 56, 57, 58, 61, 62, 63, 65, 66, 67, 68, 71, 72, 73, 75, 76, 77, 78, 81, 82, 83, 85, 86, 87, 88]
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
def get_g1(current_exp, punish_val=-1, tiny_pr_val=5, prize_val=20):
    graph1 = {
        "graph_type": {
            "bar chart": True,
            "stacked": True,
            "num vars per x": 6
        },
        "where": { # find trial nos where
            "punishment": {
                        "key_list": ["Environment", "punishment"],
                        "looking_for_value": punish_val
            },
            "tiny_prize": {
                        "key_list": ["Environment", "tiny_prize"],
                        "looking_for_value": tiny_pr_val
            },
            "prize": {
                        "key_list": ["Environment", "prize"],
                        "looking_for_value": prize_val
            }
        },
        "to_graph_X" : current_exp["test_arrays"]["traffic_probability_list"],
        "x_keywords": ["Environment", "traffic", "p"],
        "y_keywords": [ ["Results", "rewards_dict", "r1_expert"],
                        ["Results", "rewards_dict", "r2_expert"],
                        ["Results", "rewards_dict", "r1_irl"],
                        ["Results", "rewards_dict", "r2_irl"],
                        ["Results", "rewards_dict", "r1_optimal"],
                        ["Results", "rewards_dict", "r2_optimal"]
                      ],
        "xaxis_name" : "Probability of traffic",
        "yaxis_name" : "Average of collected rewards in objective values",
        "x_labels": ["Expert agent", "IRL agent", "Optimal objective agent"], # do cross product of x_labels and stack_labels
        "stack_labels": ["System 1 Rewards", "System 2 Rewards"],
        "colors": ['#908946', "#Cec464",
                   "#922f0e", "#Da816a",
                   "#58747f", "#8aafbc"]
    }
    return graph1
    # EXPERT -> dark yellow: #908946, light yellow #Cec464
    # IRL -> dark red ##922f0e light red #Da816a
    # OPTIMAL -> dark blue: #58747f, light blue #8aafbc


def get_from_dict(dictionary, list_strings):
    initial = dictionary[list_strings[0]]
    if len(list_strings)>1: initial=initial[list_strings[1]]
    if len(list_strings)>2: initial=initial[list_strings[2]]
    if len(list_strings)>3: initial = initial[list_strings[3]]
    if len(list_strings)>4: initial = initial[list_strings[4]]
    return initial


def find_trial_nos_suiting_condition(graph_dict):
    # first find trial_nos
    trial_nos = []  # which trials to extract data from for graphing
    for trial_dict in end_dicts:  # all the trial_dicts here
        add_this_one = True
        for key, nested_dict in graph_dict["where"].items(): # dict of dicts # see if all of these match
            if get_from_dict(trial_dict, nested_dict["key_list"]) != nested_dict["looking_for_value"]:  # dict of dicts # see if all of these match
                add_this_one = False
        if add_this_one: trial_nos.append(trial_dict["Experiment info"]["trial no"])

    print("trial nos for", graph_dict["where"])
    print("are", trial_nos)

    return trial_nos


def make_prob_effect_on_reward_graphs(graph_dict, save_name, save=True, deterministic=True): #make a graph for every reward of same values
    to_graph_X = graph_dict["to_graph_X"] # some test array, example prob [0.05, 0.5, 0.7, 0.95]
    trial_nos = find_trial_nos_suiting_condition(graph_dict)
    # there will only be one case mathing each probability group because other vars are constant
    groups = np.full((len(graph_dict["y_keywords"]), len(graph_dict["to_graph_X"])), None) #num probs x num bars per prob, when stacked/2
    for trial_no in trial_nos:
        this_trial_dict = end_dicts[trial_no-1] #this trial da hep 3 sey var
        print("this_trial_dict[\"Results\"][\"rewards_dict\"]", this_trial_dict["Results"]["rewards_dict"])
        print(trial_no)
        prob_here = get_from_dict(this_trial_dict, graph_dict["x_keywords"])
        prob_index = to_graph_X.index(prob_here)
        if not deterministic:
            for i, agent_type in enumerate(graph_dict["x_labels"]):
                for j, reward_type in enumerate(graph_dict["stack_labels"]):
                    index = i*len(graph_dict["stack_labels"]) + j
                    groups[index][prob_index] = get_from_dict(this_trial_dict, graph_dict["y_keywords"][index])
                    print("agent_type", agent_type, "reward_type", reward_type, "index", index)
        else:
            groups = det_trajectory_rewards(graph_dict, prob_index, this_trial_dict, groups)


    #print(groups)


    fig_chart = stacked_bar_chart(xs= graph_dict["to_graph_X"], ys = groups,
                      x_labels=graph_dict["x_labels"], stack_labels=graph_dict["stack_labels"],
                      x_axis_name=graph_dict["xaxis_name"], y_axis_name=graph_dict["yaxis_name"],
                      colors=graph_dict["colors"])

    if save:
        # Render the figure
        image_path = f"stackedbars_{save_name}.png"
        to_analysis = path_to_folder+"eval/"
        in_folder = os.path.join(to_analysis, image_path)
        pio.write_image(fig_chart, in_folder)
    else: fig_chart.show()



def stacked_bar_chart(xs, ys, x_labels, stack_labels, x_axis_name, y_axis_name, colors):

    fig = go.Figure()

    fig.update_layout(
        template="simple_white",
        xaxis=dict(title_text=x_axis_name),
        yaxis=dict(title_text=y_axis_name),
        title=save_by,
        barmode="stack",
    )

    for i, agent_type in enumerate(x_labels):
        for j, reward_type in enumerate(stack_labels):
            index = i * len(stack_labels) + j
            fig.add_trace(
                go.Bar(x=[xs, [agent_type] * len(xs)], y=ys[index], name=stack_labels[j], marker_color=colors[index]),
            )
    return fig

# prize, tiny_prize, punishment should all be same


# graphs comparing average rewards collected
# 3 lines: expert, optimal objective, irl agent

# 1st graph
#       depending on maximum reward available, when probability is constant

# 2nd graph
#       depending on traffic probability, when rewards are constant


def check_dir(dir_folders):  # dir_folders same as path_to_folder
    # Get all directories within the specified directory
    directory_names = [name for name in os.listdir(dir_folders) if os.path.isdir(os.path.join(dir_folders, name))]
    directory_names.sort()

    for i in range(len(directory_names)-2): #last 2 are always specific experiments and "other"
        # Read JSON file
        name = "all_info.json"
        f = os.path.join(os.path.join(dir_folders, directory_names[i]), name)
        with open(f, "r") as file:
            data = json.load(file)
            print("trial no", data["Experiment info"]["trial no"])
            print(directory_names[i])
    print(directory_names)

def check_file_counts(directory_names, dir_folders):  # dir_folders same as path_to_folder
    count_files = lambda directory: len([name for name in os.listdir(os.path.join(dir_folders, directory)) if
                                         os.path.isfile(os.path.join(os.path.join(dir_folders, directory), name))])
    file_nums = list(map(count_files, directory_names))
    for i in range(0, len(file_nums)-2): #last 2 are always specific experiments and "other"
        print(directory_names[i])
        print(file_nums[i])


# check_dir(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "results/"))

def det_trajectory_rewards(graph_dict, prob_index, this_trial_dict, groups):
    #make det policy
    #compute rewards from 1 traj

    irl_det = optimal_policy_from_value(get_from_dict(this_trial_dict, current_exp_setup["irl"]))
    print("irl_det")
    print(irl_det)
    print(irl_det.shape)
    for i, agent_type in enumerate(graph_dict["x_labels"]):
        for j, reward_type in enumerate(graph_dict["stack_labels"]):
            index = i * len(graph_dict["stack_labels"]) + j
            groups[index][prob_index] = get_from_dict(this_trial_dict, graph_dict["y_keywords"][index])
            print("agent_type", agent_type, "reward_type", reward_type, "index", index)
    return groups

def optimal_policy_from_value(value):
    policy = np.zeros((100, 100, 4))
    for s in range(100):
        for a in range(4):
            policy = np.argmax([value[state_index_transition(s, a)]])

    return policy

def state_point_to_index(x, y):
    return y * 10 + x

def state_point_to_index_clipped(state):
    x, y = state
    x = max(0, min(10 - 1, x))
    y = max(0, min(10 - 1, y))
    return state_point_to_index(x, y)

def state_index_to_point(state):
    x = state % 10
    y = state // 10
    return x, y

def state_index_transition(s, a):
    if s in impossible_states:
        return s
    x, y = state_index_to_point(s)
    dx, dy = actions[a]
    x += dx
    y += dy
    resulting_index = state_point_to_index_clipped((x, y))
    if resulting_index in impossible_states:
        return s
    else:
        return resulting_index



traffic_probability_list = [0.3, 0.05, 0.5, 0.7, 0.95]
prize_list = [20, 30]
tiny_prize_list = [1, 5]
punishment_list = [-1, -5, -10]

current_exp_setup = exps.exp1_bigchunk # the test is different values/combintions of punishment, prize, tiny_prize, traffic_probability

path_to_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), current_exp_setup["path"])

end_dicts = [] # list of dictionaries

# read all jsons from results in this directory, put them in array
for i in range(len(current_exp_setup["dir_names"])):
    f = os.path.join(os.path.join(path_to_folder, current_exp_setup["dir_names"][i]), "all_info.json") # to read
    with open(f, "r") as file:  # Read JSON file
        data = json.load(file)
        end_dicts.append(data)

for i in traffic_probability_list:
    for j in punishment_list:
        for p in prize_list:
            for t in tiny_prize_list:
                g = get_g1(current_exp_setup)
                g["where"] = { # find trial nos where
                        "punishment": {
                                    "key_list": ["Environment", "punishment"],
                                    "looking_for_value": j
                        },
                        "tiny_prize": {
                                    "key_list": ["Environment", "tiny_prize"],
                                    "looking_for_value": t
                        },
                        "prize": {
                                    "key_list": ["Environment", "prize"],
                                    "looking_for_value": p
                        }}

                save_by = "i = " + str(i) + " j = " + str(j) + " p = " + str(p) + " t = " + str(t)

                make_prob_effect_on_reward_graphs(g, save_name=save_by, save=True)


# CHANGE THIS TO ANALYSE OTHERS

