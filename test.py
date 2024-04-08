
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Generate sample data for x and z arrays
x = np.random.rand(400)
z = np.random.rand(1001, 400)

# Set up the figure and axis
fig, ax = plt.subplots()

# Plot the initial state of z
line, = ax.plot(z[0])

def animate(i):
    line.set_ydata(z[i])  # Update the data for the line
    return line,

# Call the animator, blit=True means only re-draw the parts that have changed.
ani = animation.FuncAnimation(fig, animate, frames=1001, interval=20, blit=True)

plt.show()