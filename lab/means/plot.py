import pandas as pd
import matplotlib.pyplot as plt
import itertools



means_df = pd.read_csv('lab/means/means.csv')
intervals_df = pd.read_csv('lab/means/ICs.csv')

means_df.plot(x='shots', title='Accuracy across different number of shots')
plt.grid()
plt.savefig('lab/means/Accuracy across different number of shots.png')
intervals_df.plot(x='shots', title='Confidence-Interval across different number of shots')
plt.grid()
plt.savefig('lab/means/Confidence-Interval across different number of shots.png')
# for column in means_df.columns[1:]:
#     plt.plot(means_df['shots'], means_df[column])
#     plt.fill_between(means_df['shots'], means_df[column] - intervals_df[column], means_df[column] + intervals_df[column], alpha=.1)

# plt.show()
