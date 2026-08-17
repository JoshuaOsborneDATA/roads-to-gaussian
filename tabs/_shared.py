import matplotlib.pyplot as plt

SEED   = 42
C_HIST = "steelblue"
C_FIT  = "#e74c3c"
C_POP  = "#e67e22"


def make_fig(ncols=2, figsize=(12, 4)):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return fig, axes
