import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

rc('text', usetex=True)
rc('text.latex',
   preamble=[
       r'\usepackage[tt=false, type1=true]{libertine}',
       r'\usepackage[libertine]{newtxmath}', r'\usepackage[varqu]{zi4}',
       r'\usepackage[T1]{fontenc}'
   ])
df = pd.read_pickle("sne.pkl")

# df = pd.read_csv('combine.csv', delimiter='\t', names = ['id', 'GCN', 'f1', 'f2', 'f3', 'f4', 'f5'])
# #print df
# #df['y'] = y
# #df['label'] = df['y'].apply(lambda i: str(i))
#
# #N = 10000
# #df_subset = df.loc[rndperm[:N],:].copy()
#
# df = df.loc[:10000,:]
# feat_cols = ['f1', 'f2', 'f3', 'f4', 'f5']
# data = df[feat_cols].values
#
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(data)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
#
# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
#
# df.to_pickle("sne.pkl")
df = df.replace('TrueSkill-similarity', 'TrueSkill')
df = df.replace('Arrival-similarity', 'Arrival')
plt.figure(figsize=(7,5))
g = sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="GCN",
    palette=sns.color_palette("hls", 4),
    data=df,
    legend="full",
    alpha=0.7
)
g.set_ylabel("")
g.set_xlabel("")
g.tick_params(direction='out', length=6, width=2, colors='k', which='major', labelsize=16)
g.tick_params(direction='out', length=4, width=1, colors='gray', which='minor')
g.set_xlim([-15, 15])
g.legend(fontsize=14, loc=4, edgecolor='white')
sns.mpl.pyplot.tight_layout()
plt.savefig('sne_plot.pdf')
