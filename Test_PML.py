import numpy as np
import matplotlib.pyplot as plt
N_x, N_y = (30,80)
PML_thickness = 15
Gradient_PML = 3

Grid = np.zeros((N_x,N_y))
print((PML_thickness*2 + Grid.shape[0], PML_thickness*2 + Grid.shape[1]))
Grid_PML = np.pad(Grid, PML_thickness, mode="linear_ramp", end_values=(PML_thickness,PML_thickness))
print(Grid_PML.shape)
Grid_PML_xn = 0.333*(Grid_PML/PML_thickness)**Gradient_PML

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

#plt.show()

