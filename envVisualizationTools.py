import plotly.graph_objects as go
import numpy as np
from rp1.gridenv import GridEnvironment

images_dict = {
    "home": "https://cdn4.iconfinder.com/data/icons/pictype-free-vector-icons/16/home-512.png",
    "donut": "https://www.google.com/url?sa=i&url=https%3A%2F%2Ficonduck.com%2Ficons%2F191290%2Fdonut&psig=AOvVaw1axzlY93rvPRlc4lPhB0mr&ust=1686506530520000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCMCvn8ekuf8CFQAAAAAdAAAAABAQ",
    "bank": "data:image/png;base64,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",
    "traffic": "https://cdn-icons-png.flaticon.com/512/55/55205.png",
    "agent": "https://cdn-icons-png.flaticon.com/512/5231/5231019.png"
}
debug = False
if debug:
    reverse_images_dict = {value: key for key, value in images_dict.items()}

    print("images_dict")
    for key, value in images_dict.items():
        value = value[8:13]  # Get only the first 5 letters of the string
        print(f'{key}: {value}')

    print("reverse_images_dict")
    for key, value in reverse_images_dict.items():
        value = value[8:5]
        print(f'{key}: {value}')

class Env_Visualization():
    def __init__(self, env, cognitive_model):
        self.env = env
        self.final_v_it = cognitive_model.value_it_1_and_2_soph_o #for now TODO
        self.initial_images = np.full((self.env.width, self.env.height), None)  # shape, fill value
        self.value_it_reshaped = np.reshape(self.final_v_it, (self.env.width, self.env.height))


    def setup_world(self):
        # row_indices, col_indices = zip(*indices)  # Converts tuples into separate lists
        for place in self.env.visual_dict["places_list"]: #place is a string
            row_indices, col_indices = zip(*list(map(self.state_index_to_point, self.env.visual_dict[place]["indices"])))
            self.initial_images[row_indices, col_indices] = images_dict[place]

    def state_point_to_index(self, x, y):
        return y * self.env.width + x
    def state_index_to_point(self, index_state):
        x = index_state % self.env.width
        y = index_state // self.env.width
        return x, y

    def make_pictured_heatmap(self):
        customdata = np.arange(0, self.env.n_states).reshape((self.value_it_reshaped.shape))
        heatmap = go.Heatmap( # Create the heatmap trace
            z=self.value_it_reshaped,
            customdata=customdata,
            colorscale='Viridis',
            hovertemplate="Accumulated value: %{z}<br> State index: %{customdata}"
            #"<br> Coordinates: %{x} %{y}"
        )

        fig = go.Figure()
        fig.add_trace(heatmap)
        self.setup_world()

        # Add images to the heatmap squares
        for i in range(self.env.width):
            for j in range(self.env.height):
                if self.initial_images[i][j]:
                    if debug: print(" coordinates " + str(i) + " " + str(j) + " index " + str(self.state_point_to_index(i, j)) + " place " + reverse_images_dict[self.initial_images[i][j]] + " link " + self.initial_images[i][j])
                    fig.add_layout_image(
                        dict(
                            source=self.initial_images[i][j],
                            xref='x',
                            yref='y',
                            x=i,
                            y=j,
                            sizex=0.9,
                            sizey=0.9,
                            xanchor='center',
                            yanchor='middle',
                            opacity=1,
                        )
                    )

        # Configure layout settings
        fig.update_layout(
            title='Simple Gridworld',
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            margin=dict(
                l=20,  # left margin
                r=20,  # right margin
                t=30,  # top margin
                b=10,  # bottom margin
            )
        )

        return fig