"""Helper functions for plotting model evaluation results."""

import numpy as np
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


def test_plot():
    """Test the plot style with a dummy plot.
    
    :return: None
    :rtype: None
    """
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    plt.plot(x, y)
    plt.savefig('test_plot.png')


def make_plot_data(scores):
    """Make data for plots.

    :param pd.DataFrame scores: scores from model evaluation
    :return: plot data
    :rtype: tuple[pd.DataFrame]
    """
    prob_true = scores.filter(regex='^test_prob_true_', axis=0)
    prob_pred = scores.filter(regex='^test_prob_pred_', axis=0)
    tn = scores.loc['test_tn', 'mean']
    fp = scores.loc['test_fp', 'mean']
    fn = scores.loc['test_fn', 'mean']
    tp = scores.loc['test_tp', 'mean']
    conf_matrix_scores = np.array([[tp, fp],
                                   [fn, tn]])
    return prob_true, prob_pred, conf_matrix_scores


def plot_calibration_errors(prob_true, prob_pred, type, save_path):
    """Plot calibration errors.
    
    :param pd.DataFrame prob_true: true probabilities
    :param pd.DataFrame prob_pred: predicted probabilities
    :param str type: model type
    :param str save_path: path to save plot
    :return: None
    :rtype: None
    """
    fold_names = [col for col in prob_true.columns
                  if col.startswith('fold')]
    num_of_folds = len(fold_names)
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            sharey=True,
                            figsize=(9, 4))
    colors = plt.get_cmap('hellafresh')
    for i in [0, 1]:
        axs[i].plot(
            [0, 1],
            [0, 1],
            linestyle='--',
            color='black',
            alpha=0.5,
            label='Perfect Calibration')
    for i, col in enumerate(fold_names):
        axs[0].plot(
            prob_pred[col],
            prob_true[col],
            color=colors(i / num_of_folds),
            label=f"{col}")
    axs[1].errorbar(
        prob_pred['mean'],
        prob_true['mean'],
        xerr=prob_pred['std'],
        yerr=prob_true['std'],
        fmt='none',
        label='Model Calibration')
    for i in [0, 1]:
        axs[i].set_xlim([0.1, 0.9])
        axs[i].set_ylim([0.1, 0.9])
        axs[i].set_title(f'Calibration Curve ({type})')
        axs[i].set_xlabel('Predicted Probability')
        axs[0].set_ylabel('True Probability')
        axs[i].legend(loc='upper left', framealpha=0.0)
    fig.tight_layout()
    fig.savefig(f"{save_path}/calibration_curve_{type}.png")
    return


def plot_confusion_matrix(scores, type, save_path):
    """Plot normalized confusion matrix using matplotlib.
    
    :param np.array scores: confusion matrix scores
    :param str type: model type
    :param str save_path: path to save plot
    :return: None
    :rtype: None
    """
    scores = (scores / scores.sum()).round(3)
    fig, ax = plt.subplots(nrows=1, ncols=1,
                           figsize=(4, 4))
    ax.imshow(scores, cmap='hellafresh')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Positive', 'Negative'])
    ax.set_yticklabels(['Positive', 'Negative'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    for i in [0, 1]:
        for j in [0, 1]:
            ax.text(j, i, scores[i, j],
                    ha='center', va='center',
                    color='xkcd:off white',
                    fontsize=18)
    ax.set_title(f'Confusion Matrix ({type})')
    fig.tight_layout()
    fig.savefig(f"{save_path}/confusion_matrix_{type}.png")
    return


def make_and_save_plots(scores, type, save_path):
    """Make and save plots.
    
    :param pd.DataFrame scores: scores from model evaluation
    :param str type: model type
    :param str save_path: path to save plot
    :return: None
    :rtype: None
    """
    prob_true, prob_pred, conf_matrix_scores = make_plot_data(scores)
    plot_calibration_errors(prob_true, prob_pred, type, save_path)
    plot_confusion_matrix(conf_matrix_scores, type, save_path)
    return


