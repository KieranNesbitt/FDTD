import numpy as np
import matplotlib.pyplot as plt
N_x, N_y = (10,10)
PML_thickness = 5
Grid = np.zeros((N_x,N_y))
def add_outer_layer(array, value, thickness):
    # Get the shape of the original array
    shape = array.shape
    # Calculate the new shape of the array including the outer layer
    new_shape = tuple(dim + 2 * thickness for dim in shape)
    
    # Create a new array with the new shape and fill it with the specified value
    new_array = np.full(new_shape, value)
    
    # Calculate the slicing indices for copying the original array into the new array
    slices = tuple(slice(thickness, dim + thickness) for dim in shape)
    # Copy the original array into the center of the new array
    new_array[slices] = array
    
    return new_array

#Grid_PML = add_outer_layer(Grid, value=1, thickness=2)

Grid_PML = np.pad(Grid, 5, mode="linear_ramp", end_values=(PML_thickness,PML_thickness))

plt.figure(2)
plt.imshow(Grid_PML, cmap="Blues")
plt.show()

