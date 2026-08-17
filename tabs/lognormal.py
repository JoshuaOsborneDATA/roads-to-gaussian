import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ._shared import SEED, C_HIST, C_FIT, make_fig


def render():
    st.header("Log-normal to Gaussian (Multiplicative CLT)")
    st.markdown(
        r"""
        A log-normal distribution arises from a **multiplicative** process. If
        $X = Y_1 \cdot Y_2 \cdots Y_n$ where each $Y_i > 0$ is an independent factor, then
        """
    )
    st.latex(r"\log X = \log Y_1 + \log Y_2 + \cdots + \log Y_n")
    st.markdown(
        r"""
        is a sum of independent terms. By the CLT this sum converges to a Gaussian,
        so $\log X \sim \mathcal{N}(\mu, \sigma^2)$ — making $X$ log-normal.
        The left panel shows the skewed raw distribution; the right panel shows that
        taking the log recovers a Gaussian.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        mu_ln = st.slider("μ (log-space mean)", -2.0, 2.0, 0.0, 0.1, key="lnmu")
    with col2:
        sig_ln = st.slider("σ (log-space std)", 0.1, 2.0, 0.5, 0.1, key="lnsig")

    rng         = np.random.default_rng(SEED)
    samples     = rng.lognormal(mu_ln, sig_ln, 5000)
    log_samples = np.log(samples)

    x_ln   = np.linspace(0.001, np.percentile(samples, 99.5), 400)
    x_norm = np.linspace(log_samples.mean() - 4 * log_samples.std(),
                         log_samples.mean() + 4 * log_samples.std(), 400)

    fig, axes = make_fig()

    axes[0].hist(samples, bins=60, density=True, color=C_HIST, alpha=0.6, label="Samples")
    axes[0].plot(x_ln, stats.lognorm.pdf(x_ln, s=sig_ln, scale=np.exp(mu_ln)),
                 color=C_FIT, lw=2, label="Log-normal PDF")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"X ~ LogNormal(μ={mu_ln:.1f}, σ={sig_ln:.1f}) — skewed")
    axes[0].legend()

    axes[1].hist(log_samples, bins=60, density=True, color=C_HIST, alpha=0.6,
                 label="log(X) samples")
    axes[1].plot(x_norm, stats.norm.pdf(x_norm, mu_ln, sig_ln),
                 color=C_FIT, lw=2, label=f"N({mu_ln:.1f}, {sig_ln:.1f}²)")
    axes[1].set_xlabel("log(X)")
    axes[1].set_title("log(X) ~ Normal — Gaussian")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why does this happen?"):
        st.markdown(
            r"""
            Taking the logarithm converts multiplication into addition. If quantities
            compound multiplicatively (prices, growth rates, physical measurements that
            combine through products), their log is a sum of independent terms. The CLT
            then guarantees the log is approximately Gaussian even if the individual
            factors are far from normal. This is why stock returns are often modelled as
            log-normal: each day's return is a multiplicative factor, and the log of the
            cumulative return is a running sum that converges to a Gaussian.
            """
        )
