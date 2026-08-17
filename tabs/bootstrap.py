import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ._shared import SEED, C_HIST, C_FIT, C_POP, make_fig


@st.cache_data
def _precompute(sample_size=100):
    rng = np.random.default_rng(SEED)
    population = np.concatenate([
        rng.normal(-4, 0.8, 3000),
        rng.normal( 0, 0.6, 4000),
        rng.normal( 4, 0.9, 3000),
    ])
    boot_means = np.array([
        rng.choice(population, size=sample_size, replace=True).mean()
        for _ in range(3000)
    ])
    return population, boot_means


def render():
    st.header("Bootstrap to Gaussian")
    st.markdown(
        r"""
        Bootstrapping repeatedly resamples from observed data to estimate the
        sampling distribution of a statistic. The distribution of bootstrap
        **sample means** converges to a Gaussian regardless of the shape of the
        original population — a direct consequence of the CLT. The population
        below is wildly non-normal (three separated modes), yet the bootstrap
        means form a bell curve.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        n_boot = st.slider("Number of bootstrap samples", 10, 3000, 50, step=10,
                           key="nboot")
    with col2:
        bs_size = st.slider("Resample size", 10, 200, 100, step=10, key="bssize")

    population, boot_means_all = _precompute(sample_size=bs_size)
    boot_means = boot_means_all[:n_boot]

    fig, axes = make_fig()

    x_pop = np.linspace(population.min(), population.max(), 400)
    kde   = stats.gaussian_kde(population)
    axes[0].fill_between(x_pop, kde(x_pop), alpha=0.5, color=C_POP)
    axes[0].plot(x_pop, kde(x_pop), color=C_POP, lw=2)
    axes[0].set_title("Original population (multimodal)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")

    axes[1].hist(boot_means, bins=min(50, max(5, n_boot // 5)),
                 density=True, color=C_HIST, alpha=0.7,
                 label=f"Bootstrap means (n={n_boot})")
    if n_boot >= 20:
        mu_b, sig_b = boot_means.mean(), boot_means.std()
        x_fit = np.linspace(mu_b - 4 * sig_b, mu_b + 4 * sig_b, 400)
        axes[1].plot(x_fit, stats.norm.pdf(x_fit, mu_b, sig_b),
                     color=C_FIT, lw=2, label="Normal fit")
    axes[1].set_title("Distribution of bootstrap sample means")
    axes[1].set_xlabel("Sample mean")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why does this happen?"):
        st.markdown(
            r"""
            Each bootstrap sample mean is the average of $n$ draws from the population.
            By the CLT, the mean of $n$ independent draws converges to a Gaussian with
            mean equal to the population mean and standard deviation $\sigma/\sqrt{n}$,
            where $\sigma$ is the population standard deviation. The shape of the original
            population is irrelevant — what matters is that the statistic is an average.
            Bootstrapping exploits this to estimate uncertainty without needing to know
            the population distribution in advance.
            """
        )
