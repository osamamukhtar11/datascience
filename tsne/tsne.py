import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_mldata
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
import os
import time
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")
from PIL import Image

def compressAndSave(file, verbose=False):
    '''
    :param file: name of image file you want to save
    :param verbose: optional explanations are printed in console if this is True
    :return:
    '''
    filepath = os.path.join(current_directory, file)
    plt.savefig(filepath)
    oldsize = os.stat(filepath).st_size
    picture = Image.open(filepath)
    picture.save(file, "JPEG", optimize=True, quality=50)
    newsize = os.stat(os.path.join(os.getcwd(), file)).st_size
    percent = (oldsize - newsize) / float(oldsize) * 100
    if (verbose):
        print(file+" compressed from {0} to {1} or {2}%".format(oldsize, newsize, percent))

def tSNE(dataset):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(dataset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    return  tsne_results

current_directory = os.path.dirname(os.path.abspath(__file__))
mnist = fetch_openml(name="mnist_784")
# with old versions, fetch_mldata is faster when .mat file is available locally
# mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.0
y = mnist.target
y = mnist['target'].astype(np.float)
print(X.shape, y.shape)

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

plt.gray()
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
#plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
fig = plt.figure( figsize=(16,7) )
for i in range(0,15):
    ax = fig.add_subplot(3,5,i+1)#, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
    ax.title.set_text("Digit: {}".format(str(df.loc[rndperm[i],'label'])))
    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
    ax.tick_params(labelcolor='w')
plt.subplots_adjust(hspace=0.3)
compressAndSave('samples.jpg', verbose=True)

# since t-SNE is computationally heavy, we work with only 10000 samples
# computing pca for comparison with 10000 samples
N = 10000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1]
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

tsne_results = tSNE(data_subset)
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
)
compressAndSave('tsne-scatterplot-2d.jpg', verbose=True)

# draw side by side 2d scatterplots for pca and tsne
plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
compressAndSave('pca-vs-tsne-scatterplot-2d.jpg', verbose=True)

# following the recommendation, and using PCA to reduce dimensions to 50, and then using t-SNE on 50 dimensional data
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
# 83% variation explained by 50 components
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
tsne_results_50pca = tSNE(pca_result_50)

df_subset['tsne-pca50-one'] = tsne_results_50pca[:,0]
df_subset['tsne-pca50-two'] = tsne_results_50pca[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)
compressAndSave('pca-2d-vs-tsne-2d-vs-tsne-pca50-2d.jpg', verbose=True)