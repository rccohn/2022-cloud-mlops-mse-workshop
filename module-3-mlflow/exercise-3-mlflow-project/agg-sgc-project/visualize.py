import matplotlib.pyplot as plt
import numpy as np


def agg_cm(cmlist):
    """
    Display confusion matrix for train/validation/test performance and save to disk.

    Parameters
    ----------
    cmlist: list-like, or dict
        list-like structure containing confusion matrices in sklearn format
        (sklearn.metrics.confusion_matrix(y_true, y_predict)
         for the training, validation, and [optional test] sets respectively.
         if dict, keys should be ('train','val','test') and values should
         be the corresponding confusion matrix

    Returns
    ---------
    fig: matplotlib Figure
        Formatted confusion matrix figure
    """
    
    if type(cmlist) == dict:
        cmlist = [cmlist["train"], cmlist["val"], cmlist.get("test", None)]
        cmlist = [
            x for x in cmlist if x is not None
        ]  # remove test if it is not in dict
    subsets = ["Train", "Valid", "Test"][: len(cmlist)]
    fig, ax = plt.subplots(
        1,
        len(cmlist),
        sharey=True,
        dpi=300,
        facecolor="w",
        figsize=(4 / 3 * len(cmlist), 2),
    )
    a = ax[0]
    a.set_yticks([0, 1])
    a.set_yticklabels(["AGG", "NGG"], fontsize=8)
    a.set_ylabel("True value", fontsize=8)
    for a, cm, title in zip(ax, cmlist, subsets):
        a.axis([-0.5, 1.5, -0.5, 1.5])
        a.set_aspect(1)
        # a.plot([0.5,0.5],[-0.5,1.5], '-k', linewidth=1)
        # a.plot([-0.5,1.5],[0.5,0.5], '-k', linewidth=1)
        a.set_xticks([0, 1])
        a.set_xticks([0.5], minor=True)
        a.set_yticks([0.5], minor=True)
        a.set_xticklabels(["NGG", "AGG"], fontsize=8)
        a.set_xlabel("Predicted Value", fontsize=8)
        a.set_title(title, fontsize=8)
        a.grid(which="minor", color="0")
        for (i, j), z in np.ndenumerate(cm):
            a.text(
                j,
                1 - i,
                "{:^5}".format(z),
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round", facecolor="w", edgecolor="0", linewidth=0.75
                ),
            )
    fig.tight_layout()

    return fig
