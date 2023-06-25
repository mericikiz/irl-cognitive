import json
import os
import rp1.evaluation_utils.exp_result_pointers as exps
import plotly.graph_objects as px
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
import plotly.io as pio
import time
from pathlib import Path
import rp1.helpers.solver_modified as S

'''
exp1_bigchunk:
    - the test is different values/combintions of punishment, prize, tiny_prize, traffic_probability
    
'''


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
        "x_labels": ["Expert agent", "IRL agent", "Optimal agent"], # do cross product of x_labels and stack_labels
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
    #trial_nos = [3001, 3002, 3003, 3004]
    #print("trial_nos in question", trial_nos)
    return trial_nos


def make_prob_effect_on_reward_graphs(graph_dict, save_name, save=True, deterministic=True): #make a graph for every reward of same values
    to_graph_X = graph_dict["to_graph_X"] # some test array, example prob [0.05, 0.5, 0.7, 0.95]
    trial_nos = find_trial_nos_suiting_condition(graph_dict)
    # there will only be one case mathing each probability group because other vars are constant
    groups = np.full((len(graph_dict["y_keywords"]), len(graph_dict["to_graph_X"])), None) #num probs x num bars per prob, when stacked/2
    trial_nos = [3000, 3001, 3002, 3003, 3004]
    for trial_no in trial_nos:
        this_trial_dict = end_dicts[trial_no-3000] #this trial da hep 3 sey var
        prob_here = get_from_dict(this_trial_dict, graph_dict["x_keywords"])
        prob_index = to_graph_X.index(prob_here)
        sumopt1, sumopt2 = det_trajectory_opt_rewards(this_trial_dict) # prob index??
        print("trial_no", trial_no, "prob_here", prob_here, "prob_index", prob_index)
        for i, agent_type in enumerate(graph_dict["x_labels"]):
            for j, reward_type in enumerate(graph_dict["stack_labels"]):
                index = i*len(graph_dict["stack_labels"]) + j
                if i < 4 or not deterministic:
                    groups[index][prob_index] = get_from_dict(this_trial_dict, graph_dict["y_keywords"][index])
                else: #i>4
                    if index==4:
                        groups[index][prob_index] = sumopt1
                    if index==5:
                        groups[index][prob_index] = sumopt2
                #print("agent_type", agent_type, "reward_type", reward_type, "index", index)
    print("groups", groups)



    fig_chart = stacked_bar_chart(xs= graph_dict["to_graph_X"], ys = groups,
                      x_labels=graph_dict["x_labels"], stack_labels=graph_dict["stack_labels"],
                      x_axis_name=graph_dict["xaxis_name"], y_axis_name=graph_dict["yaxis_name"],
                      colors=graph_dict["colors"])

    if save:
        # Render the figure
        image_path = f"{save_name}.png"
        in_folder = os.path.join(save_path, image_path)
        pio.write_image(fig_chart, in_folder)
    else: fig_chart.show()


def stacked_bar_chart(xs, ys, x_labels, stack_labels, x_axis_name, y_axis_name, colors):

    fig = go.Figure()

    # fig.update_layout(
    #     template="simple_white",
    #     xaxis=dict(title_text=x_axis_name),
    #     yaxis=dict(title_text=y_axis_name),
    #     title=save_by,
    #     barmode="stack",
    # )

    fig.update_layout(
        template="simple_white",
        xaxis=dict(title_text=x_axis_name),
        yaxis=dict(title_text="Average collected rewards"),
        title="Comparison of collected objective rewards per agent",
        barmode="stack",
    )


    for i, agent_type in enumerate(x_labels):
        index_r1 = i * len(stack_labels) + 0
        index_r2 = i * len(stack_labels) + 1
        y1 = ys[index_r1]
        y2 = ys[index_r2]
        #bases1 = [0 if y1[i] * y2[i] < 0 else min(y1[i], y2[i]) for i in range(len(y1))]
        bases2 = [0 if y1[h] * y2[h] < 0 else y1[h] for h in range(len(y1))]
        sums = y1+y2
        print("sums", sums)
        thickness = (np.ones(len(xs))).tolist()

        fig.add_trace(
            go.Bar(x=[xs, [agent_type] * len(xs)], y=y1, name="System 1 "+ agent_type, marker_color=colors[index_r1], base=0),
        )
        #name=stack_labels[0]
        fig.add_trace(
            go.Bar(x=[xs, [agent_type] * len(xs)], y=y2, name="System 2 " + agent_type, marker_color=colors[index_r2],
                   base=bases2),
        )
        fig.add_trace(
            go.Bar(x=[xs, [agent_type] * len(xs)], y=thickness, name="Total reward in category", marker_color='#000000',
                   base=sums-1),
            # go.Scatter(
            #     x=[xs, [agent_type] * len(xs)],  # Set x values to cover the entire x-axis range
            #     y=[sums, sums],  # Set the same y-value for both points
            #     mode='lines',
            #     line=dict(color='black', width=10),
            # )
        )
    #sum_lines = [y1[h] + y2[h] for h in range(len(y1))]

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
    probs = []
    descs = []
    nos = []

    for i in range(len(directory_names)): #last 2 are always specific experiments and "other"
        # Read JSON file
        name = "all_info.json"
        print("i", i)
        f = os.path.join(os.path.join(dir_folders, directory_names[i]), name)
        with open(f, "r") as file:
            data = json.load(file)
            trial_no= data["Experiment info"]["trial no"]
            if (current_exp_setup["dir_names"][i] != directory_names[i]):
                print(trial_no)
                print(current_exp_setup["dir_names"][i])
                print(directory_names[i])



    print(probs)
    print(descs)
    print(nos)
    print("lengths")
    print(len(directory_names))
    print(len(current_exp_setup["dir_names"]))




def check_file_counts(directory_names, dir_folders):  # dir_folders same as path_to_folder
    count_files = lambda directory: len([name for name in os.listdir(os.path.join(dir_folders, directory)) if
                                         os.path.isfile(os.path.join(os.path.join(dir_folders, directory), name))])
    file_nums = list(map(count_files, directory_names))
    for i in range(0, len(file_nums)-2): #last 2 are always specific experiments and "other"
        print(directory_names[i])
        print(file_nums[i])


def det_trajectory_opt_rewards(this_trial_dict):
    #make det policy
    #compute rewards from 1 traj
    rp1_obj = this_trial_dict["Cognitive Calculations"]["simple_rp_1"]
    r2_obj = this_trial_dict["Cognitive Calculations"]["reward_arr2_o"]
    #if 'simple_rp_1' in this_trial_dict["Extra"]:
    #    S.optimal_policy()
    #else:
    opt_det_v_it = get_from_dict(this_trial_dict, current_exp_setup["opt_v"])
    opt_sum_r1, opt_sum_r2 = optimal_det_policy_rewards_sum(opt_det_v_it, rp1_obj, r2_obj)

    return opt_sum_r1, opt_sum_r2

def optimal_det_policy_rewards_sum(value_it, r1, r2):
    policy = np.array([
        np.argmax([value_it[state_index_transition(s, a, current_exp_setup["impossible_states"])] for a in range(len(current_exp_setup["actions"]))])
        for s in range(current_exp_setup["n_states"])
    ])
    r1_sum = 0
    r2_sum = 0
    for state, action in enumerate(policy):
        action_index = int(action)
        r1_sum += r1[state]
        next_state = state_index_transition(state, action, current_exp_setup["impossible_states"])         # Calculate the new state based on the action
        if next_state >= 0 and next_state < len(r1): # Check if the new state is within the gridworld bounds
            r2_sum += r2[state]

    return r1_sum, r2_sum


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


def state_index_transition(s, a, impossible_states):
    if s in impossible_states:
        return s
    x, y = state_index_to_point(s)
    dx, dy = current_exp_setup["actions"][a]
    x += dx
    y += dy
    resulting_index = state_point_to_index_clipped((x, y))
    if resulting_index in impossible_states:
        return s
    else:
        return resulting_index




current_exp_setup = exps.exp1_bigchunk # the test is different values/combintions of punishment, prize, tiny_prize, traffic_probability

timeStr = time.strftime('%Y%m%d-%H%M%S') #later save inside experiment folders too
save_path = Path(current_exp_setup["results_path"], timeStr)
if not save_path.exists():
    save_path.mkdir(parents=True)

path_to_folder = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), current_exp_setup["path"])

end_dicts = [] # list of dictionaries

# read all jsons from evaluation_results in this directory, put them in array
for d in range(len(current_exp_setup["late_dir_names"])):
    f = os.path.join(os.path.join(path_to_folder, current_exp_setup["late_dir_names"][d]), "all_info.json") # to read
    with open(f, "r") as file:  # Read JSON file
        data = json.load(file)
        end_dicts.append(data)
        #if data["Environment"]["traffic"]["p"] == 0.3:
        #    print("trial no")
        #    print(data["Experiment info"]["trial no"])
        #print(data["Experiment info"]["what is being tested"])
        #print("traffic prob", data["Environment"]["traffic"]["p"])

#save_by = "Values for " + " punishment = " + str(g["where"]["punishment"]) + " prize = " \
#          + str(g["where"]["prize"]) + " tiny prize = " + str(g["where"]["tiny_prize"])


traffic_probability_list = current_exp_setup["test_arrays"]["traffic_probability_list"]
prize_list = current_exp_setup["test_arrays"]["prize_list"]
tiny_prize_list = current_exp_setup["test_arrays"]["tiny_prize_list"]
punishment_list = current_exp_setup["test_arrays"]["punishment_list"]
#for k in traffic_probability_list:
# for j in punishment_list:
#     for p in prize_list:
#         for t in tiny_prize_list:
g = get_g1(current_exp_setup)
g["where"] = { # find trial nos where
        "punishment": {
                    "key_list": ["Environment", "punishment"],
                    "looking_for_value": -5
        },
        "tiny_prize": {
                    "key_list": ["Environment", "tiny_prize"],
                    "looking_for_value": 5
        },
        "prize": {
                    "key_list": ["Environment", "prize"],
                    "looking_for_value": 30,
        },
        # "trial_no": {
        #             "key_list": ["Experiment info"]["trial no"],
        #             "looking_for_value":
        # },
        }
deterministic = True
if deterministic: name = " optimal det policy"
else: name = " optimal st policy"
# save_by ="Rewards when punishment = " + str(j) + "prize = " + str(p) + "tiny prize = " + str(t) + name
save_by ="Rewards when punishment = " + str(-5) + "prize = " + str(30) + "tiny prize = " + str(5) + name
print(save_by)
make_prob_effect_on_reward_graphs(g, save_name=save_by, save=True, deterministic=deterministic)


