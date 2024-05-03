import numpy as np
import matplotlib.pyplot as plt

class PlanoConvexLens:
    def __init__(self, a: int, b: int, r: int, thickness: int):
        self.a, self.b, self.r, self.thickness = a, b, r, thickness

    def create(self, map_: np.ndarray, value: float):
        height, width = map_.shape
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        distances = np.sqrt((x_grid - self.a)**2 + (y_grid - self.b)**2)
        mask_outer = distances > self.r
        mask_inner = distances < (self.r + self.thickness)
        map_[(mask_outer) & (mask_inner)] = value

# Example values for the plano-convex lens
a = 50  # x-coordinate of the center of the lens
b = 50  # y-coordinate of the center of the lens
r = 30  # Radius of curvature of the convex side
thickness = 10  # Thickness of the lens

# Create a blank map (2D array) to draw the plano-convex lens
map_shape = (100, 100)  # Example map shape
map_ = np.zeros(map_shape)

# Create an instance of the PlanoConvexLens class
plano_convex_lens = PlanoConvexLens(a, b, r, thickness)

# Draw the plano-convex lens shape on the map
plano_convex_lens.create(map_, value=1)

print(np.arange(0,501*1.5e-05, 1.5e-05).shape)
