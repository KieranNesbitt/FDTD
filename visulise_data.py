import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class visulise_data:
    def __init__(self):
        df_E = pd.read_csv("E_field", header=None)
        df_H = pd.read_csv("H_field", header = None)
        array_2d_E = df_E.values[:, :-1].astype(float)
        array_2d_H = df_H.values[:, :-1].astype(float)

        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(array_2d_E[100])
        axs[0].set_ylabel("$E_x$")
        axs[1].plot(array_2d_H[100])
        axs[1].set_ylabel("$H_y$")
        
        """plt.imshow(array_2d, cmap='viridis', aspect='auto')
        plt.colorbar()  
        plt.xlabel('Column Index')  
        plt.ylabel('Row Index')  """
        plt.show()
        
if __name__ == "__main__":
    visulise_data()