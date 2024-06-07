import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt 
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')


def read_data():
    """
    Reads password data from a CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing columns 'password' and 'strength' from the CSV file.
    """
    df = pd.read_csv('./data/passwords.csv', usecols=['password', 'strength'])

    return df


def make_histogram():
    """
    Generates a histogram of password strength frequencies from the password data.

    Returns:
        str: A base64-encoded PNG image of the histogram, which can be embedded directly into HTML.
    """
    df = read_data()

    plt.figure(figsize=(4, 5))
    counts, bins, rectangles = plt.hist(df['strength'], bins=[-0.5, 0.5, 1.5, 2.5], edgecolor='black', color='#007bff', alpha=0.3, rwidth=0.75)

    # add labels to bars 
    for count, rectangle in zip(counts, rectangles):
        plt.annotate(f'{int(count)}', xy=(rectangle.get_x() + rectangle.get_width() / 2, count), xytext=(0,5), textcoords='offset points', ha='center', va='bottom')

    plt.xlabel('Strength', fontweight='bold', fontsize=11, labelpad=20)
    plt.ylabel('Number of Passwords', fontweight='bold', fontsize=11, labelpad=20)
    plt.title('Distribution of Password Strengths', fontweight='bold', fontsize=12, pad=20)
    plt.xticks(ticks=[0, 1, 2], labels=['Weak (0)', 'Medium (1)', 'Strong (2)'])
    plt.tick_params(axis='both', which='major', labelsize=9, labelcolor='grey')

    plt.margins(x=0.05, y=0.2)

    # plt.show()
    # return graph as base64 image 
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64


def plot_prediction_strength(prediction):
    """
    Generates a graphical representation of a predicted password strength on a gradient scale.

    Args:
        prediction (int): The predicted strength of the password, where 0 = Weak, 1 = Medium, and 2 = Strong.

    Returns:
        str: A base64-encoded PNG image of the password strength indicator, which can be embedded directly into HTML.
    """
    fig, ax = plt.subplots(figsize=(8, 2)) 

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    norm = plt.Normalize(0, 2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=50, pad=0.05)
   
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Weak', 'Medium', 'Strong'])
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')

    cbar.outline.set_edgecolor('none')

    cbar.ax.scatter([prediction], [0.5], color='black', s=80, edgecolor='black', zorder=5)

    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlim(0, 2)
    ax.set_xticks([])

    cbar.ax.set_ylim(-1, 2)
    cbar.ax.set_xlim(-0.1, 2.1)

    # plt.show()
    # return graph as base64 image 
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return img_base64

# testing
# predicted_strength = 2 
# plot_prediction_strength(predicted_strength)
