import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import pandas as pd

def scatterplot(x_data, y_data, x_label="X", y_label="Y", title="Scatter plot", color = "r", yscale_log=False, saveOrShow=False, filename='scatter'):
    '''
    :param x_data: variable on x-axis
    :param y_data: variable on y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param color: color for the scatter points
    :param yscale_log: use log for values on y-axis
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'scatter.jpg'
    :return:
    '''
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
    ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

    if yscale_log == True:
        ax.set_yscale('log')

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def lineplot(x_data, y_data, x_label="X", y_label="Y", title="Line plot", saveOrShow=False, filename='line'):
    '''
    :param x_data: variable on x-axis
    :param y_data: variable on y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'line.jpg'
    :return:
    '''
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    ax.plot(x_data, y_data, lw = 2, color = '#539caf', alpha = 1)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def histogram(data, bins, cumulative=False, x_label = "Value", y_label = "Frequency", title = "Histogram", saveOrShow=False, filename='histogram'):
    '''
    :param data: variable whose frequency in different bins is to be represented in the histogram
    :param bins: no. of bins
    :param cumulative: If True, then a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values. The last bin gives the total number of datapoints.
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'histogram.jpg'
    :return:
    '''
    _, ax = plt.subplots()
    ax.hist(data, bins = bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def overlaid_histogram(data1, data2, bins = 0, data1_name="X1", data1_color="#539caf", data2_name="X2", data2_color="#7663b0", x_label="Value", y_label="Frequency", title="Overlaid Histogram", saveOrShow=False, filename='overlaid_histogram'):
    # Overlay 2 histograms to compare them
    # Set the bounds for the bins so that the two distributions are fairly compared
    '''
    :param data1: variable for histogram 1
    :param data2: variable for histogram 2
    :param bins: number of bins
    :param data1_name: Label for variable 1
    :param data1_color: Colour for histogram 1
    :param data2_name: Label for variable 2
    :param data2_color: Colour for histogram 2
    :param x_label: Label on x-axis
    :param y_label: Label on y-axis
    :param title: title for the diagram
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'overlaid_histogram.jpg'
    :return:
    '''
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins

    if bins == 0:
        bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)
    else:
        bins = bins

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def barplot(x_data, y_data, error_data, x_label="", y_label="", title="Barplot", saveOrShow=False, filename='barplot'):
    '''
    :param x_data: variable on x-axis, may or may not be categorical
    :param y_data: variable on y-axis, must be numeric
    :param error_data: Error bars give a general idea of how precise a measurement is, or conversely, how far from the reported value the true (error free) value might be. If the value displayed on your barplot is the result of an aggregation (like the mean value of several data points), you may want to display error bars.
    :param x_label: Label for x-axis
    :param y_label: label for y-axis
    :param title: title for the diagram
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'overlaid_histogram.jpg'
    :return:
    '''
    _, ax = plt.subplots()
    # Draw bars, position them in the center of the tick mark on the x-axis
    ax.bar(x_data, y_data, color = '#539caf', align = 'center')
    # Draw error bars to show standard deviation, set ls to 'none'
    # to remove line between points
    ax.errorbar(x_data, y_data, yerr = error_data, color = '#297083', ls = 'none', lw = 2, capthick = 2)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def groupedbarplot(x_data, y_data_list, colors, y_data_names, x_label="", y_label="", title="Grouped Barplot", saveOrShow=False, filename='grouped_barplot'):
    '''
    :param x_data: variable on x-axis, may or may not be categorical
    :param y_data_list: list of lists where each list contains values for different groups e.g. list1 consists of scores above 90 for men and women
    :param colors: list of colors for each group
    :param y_data_names: names to be displayed for each group
    :param x_label: label on x-axis
    :param y_label: label on y-axis
    :param title: title of the diagram
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'overlaid_histogram.jpg'
    :return:
    '''
    _, ax = plt.subplots()
    # Total width for all bars at one x location
    total_width = 0.8
    # Width of each individual bar
    ind_width = total_width / len(y_data_list)
    # This centers each cluster of bars about the x tick mark
    alteration = np.arange(-(total_width/2), total_width/2, ind_width)

    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        # Move the bar to the right on the x-axis so it doesn't
        # overlap with previously drawn ones
        ax.bar(x_data + alteration[i], y_data_list[i], color = colors[i], label = y_data_names[i], width = ind_width)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def stackedbarplot(x_data, y_data_list, colors, y_data_names, x_label="", y_label="", title="Stacked Barplot", saveOrShow=False, filename='stacked_barplot'):
    '''
    :param x_data: variable on x-axis, may or may not be categorical
    :param y_data_list: list of lists where each list contains values for different groups e.g. list1 consists of scores above 90 for men and women
    :param colors: list of colors for each group
    :param y_data_names: names to be displayed for each group
    :param x_label: label on x-axis
    :param y_label: label on y-axis
    :param title: title of the diagram
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a JPG file with the filename. Default value is 'overlaid_histogram.jpg'
    :return:
    '''
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color = colors[i], align = 'center', label = y_data_names[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], color = colors[i], bottom = y_data_list[i - 1], align = 'center', label = y_data_names[i])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'upper right')
    if saveOrShow:
        plt.savefig(filename+'.jpg')
    else:
        plt.show()

def scatterplot_3d(x, y, z, labels):
    '''
    plots a 3d plot using 3d input samples and labels
    :param x: dimension 1
    :param y: dimension 2
    :param z: dimension 3
    :param labels: labels for each sample (x,y,z)
    :return:
    '''
    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=x,
        ys=y,
        zs=z,
        c=labels,
        cmap='tab10'
    )
    ax.set_xlabel('d-1')
    ax.set_ylabel('d-2')
    ax.set_zlabel('d-3')
    plt.savefig('scatterplot-3d.jpg')

def make_gif(df, starting_angle = 70, ending_angle = 210, step_size = 2, gif_parts_folder = 'gifParts', filename_sample = 'PCA_angle', gif_name = 'output.gif'):
    '''
    Create a GIF by creating multiple .PNG 3d plots each showing a different angle of graph
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

'''
References:
    https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f
    https://towardsdatascience.com/interactive-data-visualization-167ae26016e8
'''
