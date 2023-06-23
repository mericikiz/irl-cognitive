traffic_probability = [0.3] #, 0.05, 0.5, 0.7, 0.95]
prize = [20, 30]
tiny_prize = [1, 5]
punishment = [-1, -3, -5, -10]

trial_no = 49
trial_nos = []
corresponding_descriptions = []
probs = []
for i in traffic_probability:
    for j in punishment:
        for p in prize:
            for t in tiny_prize:
                trial_nos.append(trial_no)
                desc = " punishment = " + str(j) + " prize = " + str(p) + " tiny_prize = " + str(t) + " traffic_probability " + str(i)
                corresponding_descriptions.append(desc)
                trial_no+=1
                probs.append(i)

print(trial_nos)
print(corresponding_descriptions)
print(probs)
