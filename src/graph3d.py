import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("../output/processed_train.csv", header=0, index_col=0)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.set_title('3D Scatter Plot')
ax.set_xlabel('region')
ax.set_ylabel('compensated')
ax.set_zlabel('auto_brand')

ax.set_xlim(0, 10)
ax.set_ylim(0, 1000)
ax.set_zlim(0, 12)

ax.view_init(elev=12, azim=40)              # elevation and angle
ax.dist=12                                  # distance

train_labels=train_df["too_much"]
train_df['compensated'] /= 100

ax.scatter(
           train_df['region'], train_df['compensated'], train_df['auto_brand'],  # data
           c=train_labels,                            # marker colour
           marker='o',                                # marker shape
           s=20,                                      # marker size
           cmap='autumn'
           )

plt.show()                                            # render the plot