import matplotlib.pyplot as plt
import numpy as np

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
    :param filename: if saveOrShow is True, plot is saved as a PNG file with the filename. Default value is 'scatter.png'
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
        plt.savefig(filename+'.png')
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
    :param filename: if saveOrShow is True, plot is saved as a PNG file with the filename. Default value is 'line.png'
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
        plt.savefig(filename+'.png')
    else:
        plt.show()

def histogram(data, n_bins, cumulative=False, x_label = "X", y_label = "Y", title = "Histogram", saveOrShow=False, filename='histogram'):
    '''
    :param data: variable whose frequency in different bins is to be represented in the histogram
    :param n_bins: no. of bins
    :param cumulative: If True, then a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values. The last bin gives the total number of datapoints.
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a PNG file with the filename. Default value is 'histogram.png'
    :return:
    '''
    _, ax = plt.subplots()
    ax.hist(data, bins = n_bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if saveOrShow:
        plt.savefig(filename+'.png')
    else:
        plt.show()

# code inspired from https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f
