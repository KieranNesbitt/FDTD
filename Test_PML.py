import numpy as np
import matplotlib.pyplot as plt
N_x, N_y = (200,200)
PML_thickness = 50
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

Grid_PML = np.pad(Grid, PML_thickness, mode="linear_ramp", end_values=(PML_thickness,PML_thickness))
Grid_PML_xn = 0.333*(Grid_PML/PML_thickness)**3

PML_1 = Grid_PML_xn
PML_2 =(1)/(1+Grid_PML_xn)
PML_3 = (1-Grid_PML_xn)/(1+Grid_PML_xn)

plt.figure()
plt.imshow(PML_1, cmap="Blues")
plt.colorbar()
plt.figure()
plt.imshow(PML_2, cmap= "Reds")
plt.colorbar()
plt.figure()
plt.imshow(PML_3, cmap="Greens")
plt.colorbar()
plt.show()

