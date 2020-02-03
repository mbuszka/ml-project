import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# %matplotlib inline

for size in ['small']:
    for min_ratings in [1, 2, 5, 20]:
        # svd = pd.read_csv(f"svd-{size}-{min_ratings}.csv")
        cf = pd.read_csv(f"cf-{size}-{min_ratings}.csv")
        knn = pd.read_csv(f"knn-{size}-{min_ratings}.csv")
        fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=[16, 5])
        fig.suptitle(f"Data set: {size}, minimum ratings per movie: {min_ratings}")
        # axes[0].set_title("Singular Value Decomposition")
        # sns.lineplot(ax=axes[0], data=svd, x='k', y='rmse', hue='normalize', style='normalize')
        axes[1].set_title("Collaborative Filtering - k most similar")
        sns.lineplot(ax=axes[1], data=knn, x='k', y='rmse', hue='normalize', style='normalize')
        cf = cf.groupby('normalize')
        fmt = lambda x: f"{x:.2f}"
        text = np.array([cf.mean().rmse.map(fmt), cf.std().rmse.map(fmt)]).T
        axes[2].axis('off')
        axes[2].set_title('Collaborative Filtering - weighted average')
        axes[2].table(
            cellText=text,
            rowLabels=cf.count().index,
            colLabels=['Error', 'Standard Dev'],
            loc='center'
        )
plt.show()