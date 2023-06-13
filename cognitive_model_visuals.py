import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#______________COGNITIVE MODEL PARAMS STARTING VALUES_____________
time_disc_1 = 0.95
time_disc_2 = 0.95
alpha = 1.0 #α > 1: Reflects a concave value function, indicating diminishing sensitivity to gains.
beta = 1.7 #β < 1: Indicates a concave value function, suggesting diminishing sensitivity to losses.
kappa = 3.0 #κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion, resulting in a linear weighting of losses.
eta = 1.5 #η > 1: Reflects an overweighting of small probabilities and underweighting of large probabilities.
cc_constant = 2.0

lam = lambda start, stop, step: np.linspace(start, stop, int((stop - start) / step) + 1)

baseline=0
baselines = np.arange(-5, 5)
belief=0.5
range_of_rewards = np.arange(-10, 20)
time_disc_1_range = time_disc_2_range = lam(0.1, 1.0, 0.1)
alpha_range = lam(0.0, 5.0, 0.5)
beta_range = lam(0.0, 5.0, 0.5)
kappa_range = lam(0.0, 5.0, 0.5)
eta_range = lam(0.0, 5.0, 0.5)

def subjective_reward(objective_reward, baseline):  # default baseline can be overwritten
    if (objective_reward > baseline):
        return (objective_reward - baseline) ** alpha
    else:
        return ((-1) * kappa * ((baseline - objective_reward) ** beta))

def subjective_probability(objective_probability, belief):  # default belief can be overwritten
    return (objective_probability / (objective_probability + (1 - belief))) ** eta

def slider_dict(steps, pad):
    return dict(active=0, #first step that is active
                currentvalue={"prefix": "Step: "}, #current value label on the slider
                pad={"t": pad}, #the padding around the slider in pixels. {"t": 50} is 50 pixels padding on the top
                steps=steps #  list of steps in the slider, Each step is a dictionary with its own configuration, including a label and a method that specifies the action to perform when the step is selected.
            )

vectorized_func = np.vectorize(subjective_reward)
y=vectorized_func(range_of_rewards, baseline)

fig = make_subplots(rows=1, cols=1)

steps_baseline = []
for i in range(len(baselines)):
    baseline = baselines[i]
    step = dict(
        method="restyle",
        args=[{"y": [vectorized_func(range_of_rewards, baseline)]}],
        label=f"baseline {i}"
    )
    steps_baseline.append(step)

steps_alpha = []
for i in range(len(alpha_range)):
    a = alpha_range[i]
    step = dict(
        method="restyle",
        args=[{"y": [vectorized_func(range_of_rewards, baseline)]}],
        label=f"alpha {i}"
    )
    steps_alpha.append(step)


sliders = [slider_dict(steps_baseline, 50), slider_dict(steps_alpha, 100)]

steps_beta = []


fig = go.Figure(data=go.Scatter(x=range_of_rewards, y=y))  # Add a scatter trace to the figure
fig.update_layout(sliders=sliders, xaxis_range=[-10, 20], yaxis_range=[-200, 100], xaxis_title='Objective rewards', yaxis_title='Subjective Rewards')
i = 0
def update_data(trace, points, state):
    # Get the slider value
    step = state["active"]

    # Update y values based on the slider value
    y = [step, step, step, step, step]

    # Update the y data of the scatter trace
    fig.data[0].y = y

    # Update the figure
    fig.update_traces(y=[y])

# Bind the update_data function to the slider
fig.data[0].on_click(update_data)


fig.show()





