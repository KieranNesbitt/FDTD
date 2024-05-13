import numpy as np
import matplotlib.pyplot as plt
class PML:
    def __init__(self, PML_thickness: int):
        self.PML_thickness = PML_thickness
    
    def create_pad(self, Grid: np.ndarray, gradient_PML: float = 3):
        Grid_PML = np.pad(Grid, self.PML_thickness, mode="linear_ramp", end_values=(self.PML_thickness,self.PML_thickness))
        Grid_PML_xn = Grid
        if self.PML_thickness != 0:
            Grid_PML_xn = 0.25 * (Grid_PML / self.PML_thickness) ** gradient_PML
        Grid = np.pad(Grid, self.PML_thickness, mode="constant", constant_values=0)

        PML_1 = Grid_PML_xn
        PML_2 = (1) / (1 + Grid_PML_xn)
        PML_3 = (1 - Grid_PML_xn) / (1 + Grid_PML_xn)

        return Grid, (PML_1, PML_2, PML_3)

