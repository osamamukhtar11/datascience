import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_mldata
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore")
from PIL import Image
import imageio

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

# create a GIF by creating multiple .PNG 3d plots each showing a different angle of graph
def make_gif(df, starting_angle = 70, ending_angle = 210, step_size = 2, gif_parts_folder = 'gifParts', filename_sample = 'PCA_angle', gif_name = 'output.gif'):
    '''
    :param df: dataframe containing three PCA components with names 'pca-one', 'pca-two', and 'pca-three', and 'y' as output label
    :param starting_angle: Angle from which the gif should start, and first picture should be shown, recommendation is 70 degrees
    :param ending_angle: Angle where gif should end, and the last picture be shown, recommendation is 210
    :param step_size: difference between angles of two pictures
    :param gif_parts_folder: directory where all .png files are saved and then used as input for creating .gif
    :param filename_sample: prefix of file name for each part used for gif, angle is appended to this sample name.
    :param gif_name: name of output GIF file
    :return:
    '''
    df['output_values'] = pd.Categorical(df['y'])
    my_color = df['output_values'].cat.codes
    for angle in range(starting_angle, ending_angle, step_size):
        # Plot initialisation
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['pca-one'], df['pca-two'], df['pca-three'], c=my_color, cmap="Set2_r", s=60)

        # make simple, bare axis lines through space:
        xAxisLine = ((min(df['pca-one']), max(df['pca-one'])), (0, 0), (0, 0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(df['pca-two']), max(df['pca-two'])), (0, 0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0, 0), (min(df['pca-three']), max(df['pca-three'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

        ax.view_init(30, angle)

        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA MNIST Data")
        if not os.path.exists('gifParts'):
            os.mkdir(gif_parts_folder)
        filename = gif_parts_folder+ '/' + filename_sample + str(angle) + '.png'
        plt.savefig(filename, dpi=96)

        episode_frames = []
        time_per_step = 0.25
        for root, _, files in os.walk(gif_parts_folder):
            file_paths = [os.path.join(root, file) for file in files]
            # sorted by modified time
            file_paths = sorted(file_paths, key=lambda x: os.path.getmtime(x))
            episode_frames = [imageio.imread(file_path)
                              for file_path in file_paths if file_path.endswith('.png')]
        episode_frames = np.array(episode_frames)
        imageio.mimsave(gif_name, episode_frames, duration=time_per_step)

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

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df.loc[rndperm,:],
    legend="full",
    alpha=0.3
)
compressAndSave('pca-scatterplot-2d.jpg')

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm,:]["pca-one"],
    ys=df.loc[rndperm,:]["pca-two"],
    zs=df.loc[rndperm,:]["pca-three"],
    c=df.loc[rndperm,:]["y"],
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
compressAndSave('pca-scatterplot-3d.jpg')

'''
References:
    https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    https://towardsdatascience.com/interactive-data-visualization-167ae26016e8
'''
