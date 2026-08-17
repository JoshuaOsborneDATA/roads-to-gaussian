import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ._shared import C_HIST, C_FIT, make_fig


def render():
    st.header("Poisson to Gaussian")
    st.markdown(
        r"""
        The Poisson($\lambda$) distribution counts rare events in a fixed interval.
        As $\lambda$ grows, its shape approaches a Normal with the same mean and variance:
        """
    )
    st.latex(
        r"\text{Poisson}(\lambda) \;\longrightarrow\; \mathcal{N}(\lambda,\;\lambda)"
        r"\quad\text{as } \lambda \to \infty"
    )

    lam = st.slider("Rate parameter λ", 1, 200, 5, key="lam")

    half = int(np.ceil(5 * np.sqrt(lam)))
    lo   = max(0, lam - half)
    hi   = lam + half + 1
    k    = np.arange(lo, hi)
    pmf  = stats.poisson.pmf(k, lam)

    k_std = (k - lam) / np.sqrt(lam)
    x_std = np.linspace(-4, 4, 400)
    x_raw = np.linspace(k.min(), k.max(), 400)

    fig, axes = make_fig()

    axes[0].bar(k, pmf, width=1.0, color=C_HIST, alpha=0.7, label=f"Poisson({lam})")
    axes[0].plot(x_raw, stats.norm.pdf(x_raw, lam, np.sqrt(lam)),
                 color=C_FIT, lw=2, label=f"N({lam}, {lam})")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Probability")
    axes[0].set_title(f"Poisson(λ={lam})")
    axes[0].legend()

    axes[1].bar(k_std, pmf * np.sqrt(lam), width=1.0 / np.sqrt(lam),
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
            By the **additivity property** of the Poisson distribution,
            $\text{Poisson}(\lambda)$ is equal in distribution to the sum of $\lambda$
            independent $\text{Poisson}(1)$ random variables — each with mean 1 and
            variance 1. Applying the Central Limit Theorem directly:

            $$\frac{\text{Poisson}(\lambda) - \lambda}{\sqrt{\lambda}}
            \;\longrightarrow\; \mathcal{N}(0,1) \quad \text{as } \lambda \to \infty$$

            This is CLT in its standard form: a sum of $\lambda$ iid finite-variance
            terms converges to a Gaussian as the number of terms grows.
            """
        )
