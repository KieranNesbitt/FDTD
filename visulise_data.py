import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class visulise_data:
    def __init__(self):
        df_E = pd.read_csv("E_field", header=None)
        df_H = pd.read_csv("H_field", header = None)
        array_2d_E = df_E.values[:, :-1].astype(float)
        plt.plot(array_2d_E[50, :])
        
        
        """plt.imshow(array_2d, cmap='viridis', aspect='auto')
        plt.colorbar()  
        plt.xlabel('Column Index')  
        plt.ylabel('Row Index')  """
        plt.show()
        
if __name__ == "__main__":
    visulise_data()