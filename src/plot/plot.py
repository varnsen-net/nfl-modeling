"""Helper functions for plotting model evaluation results."""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from src.plot.style import PALETTES, RCPARAMS, hex_to_rgb


palettes = {k : hex_to_rgb(PALETTES[k]) for k in PALETTES.keys()}
for k in palettes.keys():
    cmap = LinearSegmentedColormap.from_list(k, palettes[k])
    mpl.colormaps.register(cmap)

sns.set_theme(
    context = 'paper',
    style="white",
    font_scale = 0.60,
    rc=RCPARAMS)


def test_plot():
    """Test the plot style with a dummy plot.
    
    :return: None
    :rtype: None
    """
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    plt.plot(x, y)
    plt.savefig('test_plot.png')
