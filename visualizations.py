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

def histogram(data, bins, cumulative=False, x_label = "Value", y_label = "Frequency", title = "Histogram", saveOrShow=False, filename='histogram'):
    '''
    :param data: variable whose frequency in different bins is to be represented in the histogram
    :param bins: no. of bins
    :param cumulative: If True, then a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values. The last bin gives the total number of datapoints.
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param saveOrShow: if true, save the plot, else show the plot
    :param filename: if saveOrShow is True, plot is saved as a PNG file with the filename. Default value is 'histogram.png'
    :return:
    '''
    _, ax = plt.subplots()
    ax.hist(data, bins = bins, cumulative = cumulative, color = '#539caf')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if saveOrShow:
        plt.savefig(filename+'.png')
    else:
        plt.show()

def overlaid_histogram(data1, data2, bins = 0, data1_name="", data1_color="#539caf", data2_name="", data2_color="#7663b0", x_label="Value", y_label="Frequency", title="Overlaid Histogram", saveOrShow=False, filename='overlaid_histogram'):
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
    :param filename: if saveOrShow is True, plot is saved as a PNG file with the filename. Default value is 'overlaid_histogram.png'
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
        plt.savefig(filename+'.png')
    else:
        plt.show()
