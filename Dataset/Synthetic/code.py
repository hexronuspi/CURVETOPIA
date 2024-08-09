import os
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib.transforms import Affine2D

constants = {
    "NUM_TRAIN": 1500*7,
    "NUM_VAL": 500*7,
    "NUM_TEST": 500*7
}

base_dir = "dataset"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

for base_class_dir in [train_dir, val_dir, test_dir]:
    os.makedirs(base_class_dir, exist_ok=True)
    for shape in ["rectangle", "circle", "straightLine", "ellipse", "roundedRectangle", "regularPolygon", "star"]:
        os.makedirs(os.path.join(base_class_dir, shape), exist_ok=True)

size_canvas = {"width": 224, "height": 224}

def normalize_color(color):
    return tuple([c / 255.0 for c in color])

class ShapesGenerator:
    def __init__(self, size_cnv, inc_time_noise, stroke_weight, list_color):
        self.size_cnv = size_cnv
        self.list_color = [normalize_color(color) for color in list_color]
        self.inc_time_noise = inc_time_noise
        self.stroke_weight = stroke_weight
    
    def init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.size_cnv["width"]/100, self.size_cnv["height"]/100))
        self.ax.set_xlim(0, self.size_cnv["width"])
        self.ax.set_ylim(0, self.size_cnv["height"])
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_facecolor((1, 1, 1))

    def save_fig(self, path):
        self.fig.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close(self.fig)

    def line_noise(self, x1, y1, x2, y2, line_color):
        self.ax.plot([x1, x2], [y1, y2], color=line_color, linewidth=self.stroke_weight)

    def rectangle_gen(self, height_shape, width_shape, rotate_deg):
        x_left = (self.size_cnv["width"] - width_shape) / 2.0
        y_top = (self.size_cnv["height"] - height_shape) / 2.0
        rect = Rectangle((x_left, y_top), width_shape, height_shape, angle=rotate_deg, 
                         edgecolor=self.list_color[int(rnd.random() * len(self.list_color))], 
                         lw=self.stroke_weight, facecolor='none')
        self.ax.add_patch(rect)
    
    def circle_gen(self, height_shape, width_shape, rotate_deg):
        x_center = float(self.size_cnv["width"]) / 2.0
        y_center = float(self.size_cnv["height"]) / 2.0
        ellipse = Ellipse((x_center, y_center), width_shape, height_shape, angle=rotate_deg, 
                          edgecolor=self.list_color[int(rnd.random() * len(self.list_color))], 
                          lw=self.stroke_weight, facecolor='none')
        self.ax.add_patch(ellipse)

    def straightLine_gen(self, x1, y1, x2, y2, rotate_deg):
        self.line_noise(x1, y1, x2, y2, self.list_color[int(rnd.random() * len(self.list_color))])

    def ellipse_gen(self, height_shape, width_shape, rotate_deg):
        self.circle_gen(height_shape, width_shape, rotate_deg)

    def roundedRectangle_gen(self, height_shape, width_shape, corner_radius, rotate_deg):
        x_left = (self.size_cnv["width"] - width_shape) / 2.0
        y_top = (self.size_cnv["height"] - height_shape) / 2.0
        rect = Rectangle((x_left, y_top), width_shape, height_shape, angle=rotate_deg, 
                         edgecolor=self.list_color[int(rnd.random() * len(self.list_color))], 
                         lw=self.stroke_weight, facecolor='none', joinstyle='round')
        self.ax.add_patch(rect)
    
    def regularPolygon_gen(self, center_x, center_y, radius, num_sides, rotate_deg):
        angle = 2 * np.pi / num_sides
        points = []
        for i in range(num_sides):
            x = center_x + radius * np.cos(angle * i)
            y = center_y + radius * np.sin(angle * i)
            points.append((x, y))
        polygon = Polygon(points, closed=True, edgecolor=self.list_color[int(rnd.random() * len(self.list_color))], 
                          lw=self.stroke_weight, facecolor='none')
        t = Affine2D().rotate_deg_around(center_x, center_y, rotate_deg)
        polygon.set_transform(t + self.ax.transData)
        self.ax.add_patch(polygon)

    def star_gen(self, center_x, center_y, radius1, radius2, num_points, rotate_deg):
        angle = 2 * np.pi / num_points
        points = []
        for i in range(num_points):
            outer_x = center_x + radius1 * np.cos(angle * i)
            outer_y = center_y + radius1 * np.sin(angle * i)
            points.append((outer_x, outer_y))
            inner_x = center_x + radius2 * np.cos(angle * i + angle / 2)
            inner_y = center_y + radius2 * np.sin(angle * i + angle / 2)
            points.append((inner_x, inner_y))
        polygon = Polygon(points, closed=True, edgecolor=self.list_color[int(rnd.random() * len(self.list_color))], 
                          lw=self.stroke_weight, facecolor='none')
        t = Affine2D().rotate_deg_around(center_x, center_y, rotate_deg)
        polygon.set_transform(t + self.ax.transData)
        self.ax.add_patch(polygon)

shape_ = ShapesGenerator(size_cnv=size_canvas,
                         inc_time_noise=0.025,
                         stroke_weight=4,
                         list_color=[(0, 0, 0), 
                                     (255, 0, 0),
                                     (160, 32, 255), 
                                     (0, 32, 255), 
                                     (0, 192, 0), 
                                     (255, 160, 16)])

img_counter = 0
toggle_bool = True
base_class_dir = train_dir
print("Please wait, currently creating a dataset...")

def draw():
    global base_class_dir
    global img_counter
    global toggle_bool
    
    if img_counter == constants["NUM_TRAIN"]:
        base_class_dir = val_dir
    if img_counter == constants["NUM_TRAIN"] + constants["NUM_VAL"]:
        base_class_dir = test_dir

    shape_.stroke_weight = int(rnd.uniform(4, 12))
    shape_.init_plot()

    shape_type = img_counter % 7  

    max_dimension = min(size_canvas["width"], size_canvas["height"]) * 0.8
    if shape_type == 0:
        min_length = int(max_dimension * rnd.uniform(0.3, 0.8))
        max_length = int(max_dimension * rnd.uniform(0.5, 1.0))
        shape_.rectangle_gen(height_shape=max_length, width_shape=min_length, rotate_deg=int(rnd.uniform(-5, 5)))
    
    elif shape_type == 1:
        max_length = int(max_dimension * rnd.uniform(0.5, 1.0))
        shape_.circle_gen(height_shape=max_length, width_shape=max_length, rotate_deg=int(rnd.uniform(-180, 180)))
    
    elif shape_type == 2:
        x1, y1 = 20, size_canvas["height"] - 20
        x2, y2 = size_canvas["width"] - 20, 20
        shape_.straightLine_gen(x1, y1, x2, y2, rotate_deg=int(rnd.uniform(-180, 180)))
    
    elif shape_type == 3:
        max_length = int(max_dimension * rnd.uniform(0.5, 1.0))
        min_length = int(max_dimension * rnd.uniform(0.3, 0.8))
        shape_.ellipse_gen(height_shape=max_length, width_shape=min_length, rotate_deg=int(rnd.uniform(-180, 180)))
    
    elif shape_type == 4:
        min_length = int(max_dimension * rnd.uniform(0.3, 0.8))
        max_length = int(max_dimension * rnd.uniform(0.5, 1.0))
        corner_radius = int(rnd.uniform(5, 20))
        shape_.roundedRectangle_gen(height_shape=max_length, width_shape=min_length, corner_radius=corner_radius, rotate_deg=int(rnd.uniform(-5, 5)))
        
    elif shape_type == 5:
        num_sides = int(rnd.uniform(3, 8))
        radius = int(max_dimension * rnd.uniform(0.3, 0.5))
        center_x, center_y = size_canvas["width"] / 2, size_canvas["height"] / 2
        shape_.regularPolygon_gen(center_x, center_y, radius, num_sides, rotate_deg=int(rnd.uniform(-180, 180)))
        
    elif shape_type == 6:
        radius1 = int(max_dimension * rnd.uniform(0.3, 0.5))
        radius2 = int(max_dimension * rnd.uniform(0.1, 0.3))
        num_points = int(rnd.uniform(5, 10))
        shape_.star_gen(center_x=size_canvas["width"]/2, center_y=size_canvas["height"]/2, radius1=radius1, radius2=radius2, num_points=num_points, rotate_deg=int(rnd.uniform(-180, 180)))

    shape_folder = os.path.join(base_class_dir, ["rectangle", "circle", "straightLine", "ellipse", "roundedRectangle", "regularPolygon", "star"][shape_type])
    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)
    
    shape_.save_fig(os.path.join(shape_folder, f"{img_counter}.jpg"))

    if img_counter == constants["NUM_TRAIN"] + constants["NUM_VAL"] + constants["NUM_TEST"] - 1:
        print("Dataset Created!")
    
    img_counter += 1
    toggle_bool = not toggle_bool

for _ in range(constants["NUM_TRAIN"] + constants["NUM_VAL"] + constants["NUM_TEST"]):
    draw()

for folder in [train_dir, val_dir, test_dir]:
    for shape in ["rectangle", "circle", "straightLine", "ellipse", "roundedRectangle", "regularPolygon", "star"]:
        path = os.path.join(folder, shape)
        num_files = len(os.listdir(path))
        print(f"{path}: {num_files} files")
