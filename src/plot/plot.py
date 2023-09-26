"""Helper functions for plotting model evaluation results."""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from src.plot.style import PALETTES, RCPARAMS, hex_to_rgb


# Register custom colormaps and set plot style
palettes = {k : hex_to_rgb(PALETTES[k]) for k in PALETTES.keys()}
for k in palettes.keys():
    cmap = LinearSegmentedColormap.from_list(k, palettes[k])
    mpl.colormaps.register(cmap)
sns.set_theme(
    context = 'paper',
    style="white",
    font_scale = 0.60,
    rc=RCPARAMS)


def make_plot_data(scores):
    """"""
    folds = scores.filter(regex='^fold_', axis=1)
    prob_true = folds.filter(regex='^test_prob_true_', axis=0)
    prob_pred = folds.filter(regex='^test_prob_pred_', axis=0)
    return prob_true, prob_pred


def make_calibration_curve(prob_true, prob_pred, type, save_path):
    """"""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.get_cmap('hellafresh')
    for i, col in enumerate(prob_pred.columns):
        ax.plot(
            prob_pred[col],
            prob_true[col],
            marker='o',
            linestyle='-',
            color=colors(i/len(prob_pred.columns)),
            label=f'Fold {col}')
    ax.set_title(f'Calibration Curve ({type})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(f"{save_path}/calibration_curve_{type}.png")
    return


def make_and_save_plots(scores, type, save_path):
    """"""
    prob_true, prob_pred = make_plot_data(scores)
    make_calibration_curve(prob_true, prob_pred, type, save_path)
    return


def test_plot():
    """Test the plot style with a dummy plot.
    
    :return: None
    :rtype: None
    """
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    plt.plot(x, y)
    plt.savefig('test_plot.png')
