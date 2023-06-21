from rp1.gridenv import GridEnvironment
import plotly.graph_objects as go
import requests
from PIL import Image
import numpy as np
from io import BytesIO



images_dict = {
    "home": "https://live.staticflickr.com/65535/52990293030_ebd595a4aa_o.png", # 500 x 500
    "donut": "https://live.staticflickr.com/65535/52990377198_021607c326.jpg", # 500 x 500
    "bank": "https://live.staticflickr.com/65535/52989319802_5d6505b724.jpg", # 500 x 500
    "agent": "https://live.staticflickr.com/65535/52990377193_3a08bb56c4.jpg", # 500 x 500
    "Hroad": "https://live.staticflickr.com/65535/52989319812_5ca7c08196_o.jpg", # 141 x 141
    "Htraffic": "https://live.staticflickr.com/65535/52989319832_9f596e404b_o.jpg", # 141 x 141
    "Vroad": "https://live.staticflickr.com/65535/52990055989_ef3e9b3a06_o.jpg", # 141 x 141
    "Vtraffic": "https://live.staticflickr.com/65535/52989918146_e4f9bcffbd_o.jpg", # 141 x 141
    "grass": "https://live.staticflickr.com/65535/52989644246_902b1fb0ba_w.jpg", # 400 x 400
}


# images_dict = {
#     "home": "", # 500 x 500
#     "donut": "https://live.staticflickr.com/65535/52989889345_a174da8312.jpg", # 500 x 500
#     "traffic": "https://live.staticflickr.com/65535/52989529451_91ff899c8c.jpg", # 500 x 500
#     "bank": "https://live.staticflickr.com/65535/52989515141_736d11b70b_m.jpg", # 231 x 218
#     "agent": "https://live.staticflickr.com/65535/52988916207_64e44dccab.jpg", # 500 x 500
#     "Hroad": "https://live.staticflickr.com/65535/52988870042_9ae33bfd21_m.jpg",
#     #"Hroad": "https://live.staticflickr.com/65535/52988870042_9ae33bfd21_m.jpg", # 240 x 129
#     #"Vroad": "https://live.staticflickr.com/65535/52989607239_5a6dde002b_m.jpg", # 129 x 240
#     "Vroad": "https://live.staticflickr.com/65535/52989607239_5a6dde002b_m.jpg",
#     "grass": "https://live.staticflickr.com/65535/52989644246_902b1fb0ba_w.jpg", # 400 x 400
# }

img_sizes_dict = {
    "home": (500, 500),
    "donut": (500, 500),
    "bank": (500, 500),
    "agent": (500, 500),
    "Hroad": (141, 141),
    "Htraffic": (141, 141),
    "Vroad":  (141, 141),
    "Vtraffic": (141, 141),
    "grass": (400, 400)
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
        self.initial_images = np.full((self.env.width, self.env.height), None)  # shape, fill value
        #self.initial_images_scales = np.full((self.env.width, self.env.height), None)
        self.road_images = np.full((self.env.width, self.env.height), None)
        #self.road_images_scales = np.full((self.env.width, self.env.height), None)
        self.cell_size = 40 #for one side
        self.heatmap_width = self.cell_size * self.env.width
        self.heatmap_height = self.cell_size * self.env.height
        self.traffic_indices = self.env.settings["Environment"]["traffic"]["indices"]


    def get_img_scale(self, image_name):
        size_x, size_y = img_sizes_dict[image_name]
        print("image_name", image_name)
        print("size_x", size_x)
        print("size_y", size_y)
        width_ratio = self.cell_size / size_x
        height_ratio = self.cell_size / size_y
        scaling_factor = min(width_ratio, height_ratio)
        # scaled_width = size_x * scaling_factor
        # scaled_height = size_y * scaling_factor
        # print(scaled_width, scaled_height)
        print("scaling_factor", scaling_factor)
        return scaling_factor

    def setup_world(self):

        # row_indices, col_indices = zip(*indices)  # Converts tuples into separate lists
        env_settings = self.env.settings["Environment"]
        for place in env_settings["places_list"]: #place is a string
            row_indices, col_indices = zip(*list(map(self.state_index_to_point, env_settings[place]["indices"])))
            self.initial_images[row_indices, col_indices] = images_dict[place]
        horizontal_roads = env_settings["roads"]["horizontal"]
        vertical_roads = env_settings["roads"]["vertical"]
        #road_img_scale = self.get_img_scale("Hroad") # all roads have same size
        #grass_img_scale = self.get_img_scale("grass") #* self.cell_size

        for h_road in horizontal_roads:
            x, y = self.state_index_to_point(h_road)
            self.road_images[x, y] = images_dict["Hroad"]
            if h_road in self.traffic_indices: self.road_images[x, y] = images_dict["Htraffic"]
            #self.road_images_scales[x, y] = road_img_scale # just scale factor
        for v_road in vertical_roads:
            x, y = self.state_index_to_point(v_road)
            self.road_images[x, y] = images_dict["Vroad"]
            if v_road in self.traffic_indices: self.road_images[x, y] = images_dict["Vtraffic"]
            #self.road_images_scales[x, y] = road_img_scale
        for grass in self.env.impossible_states:
            x, y = self.state_index_to_point(grass)
            self.road_images[x, y] = images_dict["grass"]
            #self.road_images_scales[x, y] = grass_img_scale


    def state_point_to_index(self, x, y):
        return y * self.env.width + x

    def state_index_to_point(self, index_state):
        x = index_state % self.env.width
        y = index_state // self.env.width
        return x, y

    def make_pictured_heatmap(self, final_v_it):
        value_it_reshaped = np.reshape(final_v_it, (self.env.height, self.env.width))
        customdata = np.arange(0, self.env.n_states).reshape((value_it_reshaped.shape))
        heatmap = go.Heatmap( # Create the heatmap trace
            z=value_it_reshaped,
            customdata=customdata,
            colorscale='Viridis',
            hovertemplate="Value used by expert: %{z}<br> State index: %{customdata}",
            showscale=False
            #"<br> Coordinates: %{x} %{y}"
        )
        fig = go.Figure()

        fig.add_trace(heatmap)
        # Configure layout settings
        fig.update_layout(
            #title='Final Value Iteration Used by Expert',
            #xaxis=dict(title='X-axis'),
            #yaxis=dict(title='Y-axis'),
            margin=dict(
                l=0,  # left margin
                r=0,  # right margin
                t=0,  # top margin
                b=0,  # bottom margin
            )
        )
        fig.update_layout(
            width= self.heatmap_width,
            height=self.heatmap_height,
            autosize=False,
        )
        fig.update_layout(
            xaxis=dict(domain=[0, 1.0]),
            yaxis=dict(domain=[0, 1.0]),
        )
        self.setup_world()

        # Add images to the heatmap squares
        for i in range(self.env.width):
            for j in range(self.env.height):
                fig.add_layout_image(
                    dict(
                        source=self.road_images[i][j],
                        xref='x',
                        yref='y',
                        x=i,
                        y=j,
                        sizex=1.0, #self.road_images_scales[i][j],
                        sizey=1.0, #self.road_images_scales[i][j],
                        xanchor='center',
                        yanchor='middle',
                        opacity=1,
                    )
                )
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



        return fig