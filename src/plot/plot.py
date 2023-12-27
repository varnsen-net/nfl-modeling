"""Helper functions for plotting model evaluation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.plot.style import register_colormaps, set_plot_params

register_colormaps()

def test_plot():
    """Test the plot style with a dummy plot.
    
    :return: *None*
    """
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    plt.plot(x, y)
    plt.savefig('test_plot.png')


def make_plot_data(scores):
    """Make data for plots.

    :param scores: *pd.DataFrame of shape (n_splits, n_scores)*
        Scores from model evaluation.
    :return: *tuple of dataframes*
        Plot data.
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


def plot_train_calibration(prob_true, prob_pred, name, save_path):
    """Plot calibration errors.
    
    :param prob_true: *pd.DataFrame of shape (n_splits, n_scores)*
        True probabilities.
    :param prob_pred: *pd.DataFrame of shape (n_splits, n_scores)*
        Predicted probabilities.
    :param name: *str*
        Name of model.
    :param save_path: *str*
        Path to save plots.
    :return: *None*
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
        axs[i].set_title(f'Calibration Curve ({name})')
        axs[i].set_xlabel('Predicted Probability')
        axs[0].set_ylabel('True Probability')
        axs[i].legend(loc='upper left', framealpha=0.0)
    fig.tight_layout()
    fig.savefig(f"{save_path}/{name}_calibration_curve.png")
    return


def plot_test_calibration(holdout_scores, name, save_path):
    """Plot calibration curve for holdout data.
    
    :param holdout_scores: *dict*
        Scores from model evaluation.
    :param name: *str*
        Name of model.
    :param save_path: *str*
        Path to save plots.
    :return: *None*
    """
    x = [holdout_scores[f] for f in holdout_scores if f.startswith('prob_pred_')]
    y = [holdout_scores[f] for f in holdout_scores if f.startswith('prob_true_')]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = plt.get_cmap('hellafresh')
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle='--',
        color='black',
        alpha=0.5,
        label='Perfect Calibration')
    ax.plot(x, y, label='Model Calibration')
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0.1, 0.9])
    ax.set_title(f'Calibration Curve ({name})')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.legend(loc='upper left', framealpha=0.0)
    fig.tight_layout()
    fig.savefig(f"{save_path}/{name}_calibration_curve_holdout.png")
    return


def plot_confusion_matrix(scores, name, save_path):
    """Plot normalized confusion matrix using matplotlib.
    
    :param scores: *np.array of shape (2, 2)*
        Confusion matrix scores.
    :param name: *str*
        Name of model.
    :param save_path: *str*
        Path to save plots.
    :return: *None*
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
    ax.set_title(f'Confusion Matrix ({name})')
    fig.tight_layout()
    fig.savefig(f"{save_path}/{name}_confusion_matrix.png")
    return


def make_and_save_plots(scores, name, save_path):
    """Make and save plots.
    
    :param scores: *pd.DataFrame*
        Scores from model evaluation.
    :param name: *str*
        Name of model.
    :param save_path: *str*
        Path to save plots.
    :return: *None*
    """
    set_plot_params()
    prob_true, prob_pred, conf_matrix_scores = make_plot_data(scores)
    plot_train_calibration(prob_true, prob_pred, name, save_path)
    plot_confusion_matrix(conf_matrix_scores, name, save_path)
    return


