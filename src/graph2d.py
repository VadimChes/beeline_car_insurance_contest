import pandas as pd
import pylab as plt
figsize = (10.0, 8.0)

train_df = pd.read_csv("../output/processed_train.csv", header=0, index_col=0)
train_labels=train_df["too_much"]
train_df['compensated'] /= 100

plt.scatter(train_df['auto_brand'], train_df['compensated'], c=train_labels, s=30, cmap='autumn')
plt.show()