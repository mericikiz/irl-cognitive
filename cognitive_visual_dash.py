import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

lam = lambda start, stop, step: np.linspace(start, stop, int((stop - start) / step) + 1)

range_of_rewards = lam(-10, 20, 0.1)
range_probability = lam(0.0, 1.0, 0.02)
event_prob_list = [0.1, 0.5, 0.95] #lam(0.0, 1.0, 0.2)

app = dash.Dash(__name__)

slider_configs = [
    {
        'id': 'alpha',
        'min': 0.0,
        'max': 1.5,
        'step': 0.05,
        'value': 0.7,
        'label': 'Alpha: α < 1: Reflects a concave value function, indicating diminishing sensitivity to gains',
    },
    {
        'id': 'beta',
        'min': 0.0,
        'max': 1.5,
        'step': 0.05,
        'value': 0.55,
        'label': 'Beta: β < 1: Indicates a convex value function, suggesting diminishing sensitivity to losses',
    },
    {
        'id': 'kappa',
        'min': 0.0,
        'max': 5.0,
        'step': 0.1,
        'value': 2.0,
        'label': 'Kappa: κ > 1: degree of loss aversion, higher values stronger aversion to losses., κ = 1: no loss aversion',
    },
    {
        'id': 'eta',
        'min': 0.0,
        'max': 1.5,
        'step': 0.05,
        'value': 0.8,
        'label': 'Eta: η < 1: Reflects an overweighting of small probabilities and underweighting of large probabilities',
    },
    {
        'id': 'baseline',
        'min': 0.0,
        'max': 10.0,
        'step': 1.0,
        'value': 0.0,
        'label': 'Baseline: The reward that the agent already has. Reduced sensitivity to higher gains and higher losses',
    },
    {
        'id': 'obj_prob',
        'min': 0.0,
        'max': 1.0,
        'step': 0.05,
        'value': 0.5,
        'label': 'Objective Probability of this event happening',
    },
    # {
    #     'id': 'belief',
    #     'min': 0.0,
    #     'max': 1.0,
    #     'step': 0.1,
    #     'value': 0.5,
    #     'label': 'Belief: Agent belief on how likely something is to happen',
    # },
    # { FOR NOW LEAVE TIME DISCOUNTS OUT, ALSO CC_CONSTANT
    #     'id': 'time_disc_1',
    #     'min': 0.0,
    #     'max': 1.0,
    #     'step': 0.1,
    #     'value': 0.95,
    #     'label': 'Time Discount System 1:',
    # },
    # {
    #     'id': 'time_disc_2',
    #     'min': 0.0,
    #     'max': 1.0,
    #     'step': 0.1,
    #     'value': 0.95,
    #     'label': 'Time Discount System 2:',
    # },
]
layout_options = {
    # 'title': 'My Graph',
    # 'xaxis_title': 'Objective Rewards',
    # 'yaxis_title': 'Subjective Rewards',
    'xaxis_range': [-10, -20],
    'yaxis_range': [-100, 50],
    'xaxis_fixedrange': True,
    'yaxis_fixedrange': True
}

sliders = []
for slider in slider_configs:
    sliders.append(html.Label(slider['label'], style={'marginRight': '50px', 'marginLeft': '50px'}))
    sliders.append(dcc.Slider(
        id=slider['id'],
        min=slider['min'],
        max=slider['max'],
        step=slider['step'],
        value=slider['value']
    ))


# Define the layout
app.layout = html.Div([
    html.H1("Effect of cognitive constants on reward perception"),
    *sliders,
    dcc.Graph(id='graph'),
])
@app.callback(
        Output('graph', 'figure'),
    [
        Input('alpha', 'value'),
        Input('beta', 'value'),
        Input('kappa', 'value'),
        Input('eta', 'value'),
        Input('baseline', 'value'),
        Input('obj_prob', 'value'),
        #Input('belief', 'value'),
    ]
)
def update_graph(alpha, beta, kappa, eta, baseline, obj_prob): #, belief):
    y1 = vectorized1(range_of_rewards, baseline, alpha, kappa, beta)
    y2 = vectorized2(range_probability, eta) #belief, eta)


    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Perception of Reward', 'Perception of Probability', 'Combined Perception of Final Utility'])

    trace1 = go.Scatter(x=range_of_rewards, y=y1, mode='lines', name="Subjective reward valuation")
    trace2 = go.Scatter(x=range_probability, y=y2, mode='lines', name="Subjective probability assessment")

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=2)

    for prob in event_prob_list:
        subj_prob = subjective_probability(prob, eta)  # belief, eta)
        obj_utility = prob * range_of_rewards
        y3 = y1 * subj_prob  # y1*y2 #element by element multiplication
        trace3 = go.Scatter(x=obj_utility, y=y3, mode='lines', name=("Subjective utility for event with objective probability "+str(prob)))
        fig.add_trace(trace3, row=1, col=3)

    # fig.update_layout(title_text='Perception of Reward', row=1, col=1)
    # fig.update_layout(title_text='Perception of Probability', row=1, col=2)
    # fig.update_layout(title_text='Combined Perception of Final Utility', row=1, col=3)

    fig.update_yaxes(title_text='Subjective reward', row=1, col=1)#, range=[-100, 50])  # Update x-axis title for subplot 1
    fig.update_yaxes(title_text='Decision weight', row=1, col=2)#, range=[0, 1])  # Update x-axis title for subplot 2
    fig.update_yaxes(title_text='Subjective utility', row=1, col=3)#, range=[-100, 50])  # Update x-axis title for subplot 3

    fig.update_xaxes(title_text='Objective Reward', row=1, col=1)
    fig.update_xaxes(title_text='Objective Probability', row=1, col=2)
    fig.update_xaxes(title_text='Objective Utility', row=1, col=3)


    return fig


def subjective_reward(objective_reward, baseline, alpha, kappa, beta):  # default baseline can be overwritten
    if (objective_reward > baseline):
        return (objective_reward - baseline) ** alpha
    else:
        return ((-1) * kappa * ((baseline - objective_reward) ** beta))

def subjective_probability(objective_probability, eta):  # default belief can be overwritten
    return (objective_probability**eta) / (objective_probability**eta + (1-objective_probability)**eta)
#w(p) = p^γ / (p^γ + (1 - p)^γ)



vectorized1 = np.vectorize(subjective_reward)
vectorized2 = np.vectorize(subjective_probability)




if __name__ == '__main__':
    app.run_server(debug=True)
