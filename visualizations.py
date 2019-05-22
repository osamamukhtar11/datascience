import matplotlib.pyplot as plt
import numpy as np

def scatterplot(x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
    '''
    :param x_data: variable on x-axis
    :param y_data: variable on y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
    :param color: color for the scatter points
    :param yscale_log: use log for values on y-axis
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

def lineplot(x_data, y_data, x_label="", y_label="", title=""):
    '''
    :param x_data: variable on x-axis
    :param y_data: variable on y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param title: title for the plot
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
