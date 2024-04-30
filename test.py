import numpy as np
import matplotlib.pyplot as plt

class Mask:
    def draw_circle(self,map_: np.ndarray, a:int, b:int, r:int, value: float):
        """
        Draw a filled circle onto a 2D array.

        Parameters:
            map_ (numpy.ndarray): The 2D array representing the map.
            a (int): x-coordinate of the center of the circle.
            b (int): y-coordinate of the center of the circle.
            r (int): Radius of the circle.

        Explanation:
            - Get the height and width of the map.
            - Create coordinate grids for x and y.
            - Calculate the squared distance from each point on the map to the center of the circle using the equation of a circle: (x-a)^2 + (y-b)^2.
            - Create a mask where True values indicate that the corresponding point is within the circle (i.e., the distance is less than or equal to r^2).
            - Update the map where the mask is True to fill all cells within the circle.
        Equation:
            \(Distance = (x - a)^2 + (y - b)^2)\)
        Returns:
            None (The map is modified in place).
        """
        # Get the height and width of the map
        height, width = map_.shape

        # Create coordinate grids for x and y
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Calculate the squared distance from each point to the center of the circle
        distances = (x_grid - a)**2 + (y_grid - b)**2

        # Check if the distance is less than or equal to r^2
        mask = distances <= r**2

        # Update the map where the mask is True
        map_[mask] = value

    def draw_square(self,map_, center, side_length, value):
        # Calculate the coordinates of the square's corners
        top_left = (center[0] - side_length // 2, center[1] - side_length // 2)
        bottom_right = (center[0] + side_length // 2, center[1] + side_length // 2)

        # Set the values in the map within the square's boundaries to 1
        map_[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = value


# Example usage:
import Dielectric_Mask as Draw
mask = Draw.Ellipsoid(500,500,250,50)
height, width = 1000, 1000
map_ = np.zeros((height, width), dtype=np.float32)
mask.create(map_, 1.7)
plt.imshow(map_)
plt.show()