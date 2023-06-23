import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rp1.envVisualizationTools import Env_Visualization
import rp1.irlutils as irlutils
import rp1.helpers.plot_modified as P
import rp1.helpers.solver_modified as S
import mpld3
import numpy as np

import plotly.offline as offline
import plotly.graph_objects as go
import time
from pathlib import Path
import inspect
import copy
import json
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return np.round(obj, 3).tolist()
        return JSONEncoder.default(self, obj)

plt.rcParams['figure.figsize'] = [9, 5]  # set default figure size
plt.rcParams['image.interpolation'] = 'none'
style = {  # global style for plots
    'border': {'color': 'red', 'linewidth': 0.5},
}
debug=False

many_colors = ['red', 'green', 'blue', 'orange', 'brown', "goldenrod", "magenta", 'lightpink', 'yellow',
               'darkolivegreen', "darkviolet", "turquoise", "dimgray", "cyan", "cornflowerblue", "limegreen",
               "deeppink", "palevioletred", "lavender", "bisque", "greenyellow", "honeydew", "hotpink", "indianred",
               "indigo", "ivory", "lawngreen", "lightblue", "lightgray", "lightcoral", "lightcyan", "lemonchiffon", "lightgoldenrodyellow"] #just take as many colors as you need
style = {  # global style for plots
            'border': {'color': 'red', 'linewidth': 0.5},
        }


def dump_info_to_text(save_path, settings_orijinal, cognitive_dict, results_dict):
    settings = copy.deepcopy(settings_orijinal)
    settings["Results"] = results_dict
    settings["Results"] = results_dict
    settings["Cognitive Calculations"] = cognitive_dict
    with open(str(save_path / ('all_info.json')), "w") as json_file:
        json.dump(settings, json_file, cls=NumpyArrayEncoder)


def save_matplotlib(save_path, name, fig, html=False):
    if html:
        html_fig = mpld3.fig_to_html(fig)
        with open(str(save_path / (name + '.html')), 'w') as f:
            f.write(html_fig)
    else:
        fig.savefig(str(save_path / (name + '.png')), dpi=200, bbox_inches='tight')
    plt.close(fig) #free up resources