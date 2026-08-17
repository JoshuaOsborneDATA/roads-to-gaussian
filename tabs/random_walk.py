import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ._shared import SEED, C_HIST, C_FIT, make_fig


def render():
    st.header("Random Walk to Gaussian")
    st.markdown(
        r"""
        A random walk takes steps of exactly $+1$ or $-1$ with equal probability.
        At any single step the distribution is as far from Gaussian as possible:
        two point masses. Yet as the number of steps grows, the position becomes
        approximately Normal with mean 0 and variance $n$:
        """
    )
    st.latex(r"X_1 + X_2 + \cdots + X_n \;\sim\; \mathcal{N}(0,\; n) \quad \text{as } n \to \infty")
    st.markdown(
        r"""
        The left panel shows 500 individual walk paths. The right panel shows
        the histogram of all walk positions at the chosen step with a fitted
        Normal overlay. At step 1 you see two exact spikes at $\pm 1$; as steps
        accumulate the spread grows and the bell shape emerges.
        """
    )

    step = st.slider("Step number (n)", 1, 150, 1, key="rwstep")

    rng       = np.random.default_rng(SEED)
    n_walks   = 500
    n_steps   = 150
    positions = np.cumsum(
        rng.choice([-1, 1], size=(n_walks, n_steps)).astype(float), axis=1
    )

    fig, axes = make_fig(figsize=(12, 4))

    time_ax = np.arange(1, n_steps + 1)
    for i in range(n_walks):
        axes[0].plot(time_ax[:step], positions[i, :step],
                     color="steelblue", alpha=0.05, lw=0.6)
    axes[0].axvline(step, color="#555", lw=1, alpha=0.6)
    axes[0].set_xlim(0, n_steps)
    axes[0].set_ylim(positions.min() * 1.05, positions.max() * 1.05)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Position")
    axes[0].set_title(f"500 random walks at step {step} (±1 steps)")

    pos_n  = positions[:, step - 1]
    mu_rw  = pos_n.mean()
    sig_rw = pos_n.std() if pos_n.std() > 0 else 1.0
    x_rw   = np.linspace(pos_n.min() - sig_rw, pos_n.max() + sig_rw, 400)
    axes[1].hist(pos_n, bins=35, density=True,
                 color=C_HIST, alpha=0.7, label="Walk positions")
    axes[1].plot(x_rw, stats.norm.pdf(x_rw, mu_rw, sig_rw), color=C_FIT, lw=2,
                 linestyle="--", label=f"N({mu_rw:.1f}, {sig_rw:.1f}²)")
    axes[1].set_xlabel("Position")
    axes[1].set_title(f"Positions at step {step} with Normal fit")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why does this happen?"):
        st.markdown(
            r"""
            Each position after $n$ steps is the sum of $n$ independent draws of
            $\pm 1$ — two point masses, as far from Gaussian as a distribution can be.
            Yet the CLT only requires finite mean and variance, which $\pm 1$ steps have
            (mean = 0, variance = 1). Shape is irrelevant: any sum of sufficiently many
            finite-variance terms converges to a Gaussian. The random walk is the CLT
            playing out in time.

            The spread grows as $\sqrt{n}$ — that is expected and correct. What the
            CLT guarantees is the shape, not the scale: divide by $\sqrt{n}$ and the
            result collapses onto N(0, 1) exactly.
            """
        )
