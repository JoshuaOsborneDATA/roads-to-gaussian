"""
Roads to the Gaussian
Interactive dashboard showing four ways the Normal distribution emerges.
"""

import streamlit as st
import numpy as np
from scipy import stats
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

st.set_page_config(
    page_title="Roads to the Gaussian",
    page_icon="🔔",
    layout="wide",
)

st.title("Roads to the Gaussian")
st.markdown(
    """
    The Normal distribution appears across an extraordinary range of seemingly
    unrelated situations. This dashboard demonstrates four distinct generative
    processes that all converge to a Gaussian, each with its own mechanism.
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Poisson → Gaussian",
    "Uniform Convolutions → Gaussian",
    "Log-normal → Gaussian",
    "Bootstrap → Gaussian",
])

with tab1:
    st.header("Poisson to Gaussian")
    st.markdown(
        r"""
        The Poisson($\lambda$) distribution counts rare events in a fixed interval.
        As $\lambda$ grows, its shape approaches a Normal with the same mean and variance:
        $$\text{Poisson}(\lambda) \;\longrightarrow\; \mathcal{N}(\lambda,\;\lambda)
        \quad\text{as } \lambda \to \infty$$
        """
    )

    lam = st.slider("Rate parameter λ", 1, 200, 5, key="lam")

    lo  = max(0, int(lam - 5 * np.sqrt(lam)))
    hi  = int(lam + 5 * np.sqrt(lam)) + 1
    k   = np.arange(lo, hi)
    pmf = stats.poisson.pmf(k, lam)

    k_std = (k - lam) / np.sqrt(lam)
    x_std = np.linspace(-4, 4, 400)
    x_raw = np.linspace(k.min(), k.max(), 400)

    fig, axes = make_fig()

    axes[0].bar(k, pmf, color=C_HIST, alpha=0.7, label=f"Poisson({lam})")
    axes[0].plot(x_raw, stats.norm.pdf(x_raw, lam, np.sqrt(lam)),
                 color=C_FIT, lw=2, label=f"N({lam}, {lam})")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"Poisson(λ={lam})")
    axes[0].legend()

    axes[1].bar(k_std, pmf * np.sqrt(lam), width=1 / np.sqrt(lam),
                color=C_HIST, alpha=0.7, label="Standardised Poisson")
    axes[1].plot(x_std, stats.norm.pdf(x_std), color=C_FIT, lw=2, label="N(0,1)")
    axes[1].set_xlim(-4, 4)
    axes[1].set_xlabel("(k − λ) / √λ")
    axes[1].set_title("Standardised")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why does this happen?"):
        st.markdown(
            r"""
            Poisson($\lambda$) can be thought of as the sum of $\lambda$ independent
            Bernoulli increments. By the Central Limit Theorem, the sum of many
            independent identically distributed variables converges to a Gaussian
            regardless of the original distribution. As $\lambda$ grows, more terms
            enter the sum and the bell-curve shape emerges.
            """
        )

with tab2:
    st.info("Coming soon: Uniform convolutions demo.")

with tab3:
    st.info("Coming soon: Log-normal and multiplicative CLT demo.")

with tab4:
    st.info("Coming soon: Bootstrap sampling distribution demo.")

st.caption("© Joshua Osborne")
