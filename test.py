
import numpy as np

# Example function to append E_field to a list of arrays
def append_E_field_to_list(E_field, array_list):
    # Append the E_field to the list
    array_list.append(E_field)

# Example usage:
# Create an empty list to hold the E_field data
E_field_list = []

# Example: Generate 2D numpy arrays representing E_field at different time steps
num_time_steps = 3
for step in range(num_time_steps):
    E_field = np.random.rand(200, 200)
    append_E_field_to_list(E_field, E_field_list)

# Convert the list of arrays to a 3D numpy array
E_field_array = np.array(E_field_list)

# Display the resulting 3D numpy array shape
print("Shape of the resulting 3D array:", E_field_array)