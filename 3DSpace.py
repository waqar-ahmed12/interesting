# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Sample data for a 3D line
# x = np.array([0, 1, 2, 3, 4])
# y = np.array([1, 3, 2, 4, 1])
# z = np.array([2, 4, 1, 3, 0])

# # Create a 3D figure
# fig = plt.figure(figsize=(8, 6))

# # Adjust subplot parameters to fill the figure
# ax = fig.add_subplot(111, projection='3d',  # 3D projection
#                     #left=0, right=1, bottom=0, top=1) # Subplot params (alternative)
#                     )
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Figure params

# # Plot the 3D line
# ax.plot(x, y, z, marker='o', linestyle='-', color='b', label='3D Line')

# # Remove the axes 
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# # Customize the plot
# ax.set_title('3D Line Plot')
# ax.legend()

# # Rotate the view
# ax.view_init(elev=30, azim=45)

# # Show the plot
# plt.show()







import plotly.graph_objects as go
import numpy as np

# Sample data for a 3D line
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 2, 4, 1])
z = np.array([2, 4, 1, 3, 0])

# Create the 3D line trace
trace = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers+lines',  # Show both markers and lines
    marker=dict(size=5, color='blue'),
    line=dict(color='blue'),
    name='3D Line'
)

# Create the layout
layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),  # Remove margins
    scene=dict(
        xaxis=dict(title='X Axis', showticklabels=False, showbackground=False, gridcolor='lightgrey'),
        yaxis=dict(title='Y Axis', showticklabels=False, showbackground=False, gridcolor='lightgrey'),
        zaxis=dict(title='Z Axis', showticklabels=False, showbackground=False, gridcolor='lightgrey'),
        camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)), # Adjust initial camera view
    ),
    title='3D Line Plot'
)

# Create the figure and plot
fig = go.Figure(data=[trace], layout=layout)
fig.show()  # Display the plot (interactive)