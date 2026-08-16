"""
Roads to the Gaussian
Interactive dashboard showing four ways the Normal distribution emerges,
and one famous exception that doesn't.
"""

import streamlit as st
import numpy as np
from scipy import stats
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

SEED   = 42
C_HIST = "steelblue"
C_FIT  = "#e74c3c"
C_POP  = "#e67e22"


@st.cache_data
def precompute_convolutions(n_max=20):
    dx    = 0.005
    x_u   = np.arange(0, 1 + dx * 0.5, dx)
    u_pdf = np.ones(len(x_u))
    u_pdf /= u_pdf.sum() * dx
    results = [(x_u.copy(), u_pdf.copy())]
    cur_x, cur_pdf = x_u.copy(), u_pdf.copy()
    for _ in range(n_max - 1):
        conv = fftconvolve(cur_pdf, u_pdf) * dx
        conv = np.maximum(conv, 0)
        conv /= conv.sum() * dx
        x_new = np.linspace(cur_x[0] + x_u[0], cur_x[-1] + x_u[-1], len(conv))
        results.append((x_new, conv))
        cur_x, cur_pdf = x_new, conv
    return results


@st.cache_data
def precompute_bootstrap(sample_size=100):
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Poisson → Gaussian",
    "Uniform Convolutions → Gaussian",
    "Log-normal → Gaussian",
    "Bootstrap → Gaussian",
    "Random Walk → Gaussian",
    "The Exception: Cauchy",
])

with tab1:
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
    st.header("Uniform Convolutions to Gaussian")
    st.markdown(
        r"""
        Adding $n$ independent Uniform(0, 1) random variables produces a distribution
        that evolves through convolution. After just a handful of additions the result
        is visually indistinguishable from a Gaussian:
        """
    )
    st.latex(
        r"X_1 + X_2 + \cdots + X_n \;\longrightarrow\;"
        r"\mathcal{N}\!\left(\tfrac{n}{2},\;\tfrac{n}{12}\right)"
        r"\quad\text{as } n \to \infty"
    )

    n_conv = st.slider("Number of Uniform(0,1) variables added", 1, 20, 1, key="nconv")

    conv_results     = precompute_convolutions(n_max=20)
    x_conv, pdf_conv = conv_results[n_conv - 1]
    mu_n    = n_conv / 2
    sigma_n = np.sqrt(n_conv / 12)
    x_std_conv  = (x_conv - mu_n) / sigma_n
    pdf_std_conv = pdf_conv * sigma_n
    x_gauss = np.linspace(-4, 4, 400)

    fig, axes = make_fig()

    axes[0].fill_between(x_conv, pdf_conv, alpha=0.5, color=C_HIST)
    axes[0].plot(x_conv, pdf_conv, color=C_HIST, lw=2)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Sum of {n_conv} Uniform(0,1) variable{'s' if n_conv > 1 else ''}")

    axes[1].fill_between(x_std_conv, pdf_std_conv, alpha=0.5, color=C_HIST,
                         label="Standardised sum")
    axes[1].plot(x_std_conv, pdf_std_conv, color=C_HIST, lw=2)
    axes[1].plot(x_gauss, stats.norm.pdf(x_gauss), color=C_FIT, lw=2,
                 linestyle="--", label="N(0,1)")
    axes[1].set_xlim(-4, 4)
    axes[1].set_xlabel("Standardised value")
    axes[1].set_title("Standardised — compared to N(0,1)")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why does this happen?"):
        st.markdown(
            r"""
            Convolution is the mathematical operation for adding independent random
            variables. Each time we add another Uniform we smooth the distribution
            further. The Uniform is maximally non-Gaussian (perfectly flat), yet the
            sum converges rapidly to a bell curve. By $n = 12$, the sum of Uniform(0,1)
            variables has mean 6 and variance 1 — which is why averaging 12 uniform
            draws was once used as a quick approximation for generating standard normal
            samples before efficient algorithms existed.
            """
        )

with tab3:
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

    rng_ln      = np.random.default_rng(SEED)
    samples     = rng_ln.lognormal(mu_ln, sig_ln, 5000)
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

with tab4:
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

    population, boot_means_all = precompute_bootstrap(sample_size=bs_size)
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

with tab5:
    st.header("Random Walk to Gaussian")
    st.markdown(
        r"""
        A random walk takes steps of exactly $+1$ or $-1$ with equal probability.
        At any single step the distribution is as far from Gaussian as possible:
        two point masses. Yet as the number of steps grows, the standardised
        position converges to N(0, 1). This is the CLT in a physical, path-based setting.
        """
    )
    st.latex(
        r"\frac{X_1 + X_2 + \cdots + X_n}{\sqrt{n}} \;\longrightarrow\; \mathcal{N}(0, 1)"
        r"\quad \text{as } n \to \infty"
    )
    st.markdown(
        r"""
        The left panel shows 500 individual walk paths. The right panel shows
        the histogram of all walk positions at the chosen step with a fitted
        Normal overlay. At step 1 you see two exact spikes at $\pm 1$; as steps
        accumulate the spread grows as $\sqrt{n}$ and the bell shape emerges.
        """
    )

    step = st.slider("Step number (n)", 1, 150, 1, key="rwstep")

    rng_rw   = np.random.default_rng(SEED)
    n_walks  = 500
    n_steps  = 150
    raw_steps = rng_rw.choice([-1, 1], size=(n_walks, n_steps)).astype(float)
    positions = np.cumsum(raw_steps, axis=1)

    fig, axes = make_fig(figsize=(12, 4))

    time_ax = np.arange(1, n_steps + 1)
    for i in range(n_walks):
        axes[0].plot(time_ax[:step], positions[i, :step],
                     color="steelblue", alpha=0.05, lw=0.6)
    axes[0].axvline(step, color="white" if False else "#555", lw=1, alpha=0.6)
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


with tab6:
    st.header("The Exception: Cauchy Distribution")
    st.markdown(
        r"""
        Every tab so far has shown a different road to the Gaussian. This tab shows
        a distribution that **never arrives** — no matter how many terms you sum,
        the result stays Cauchy.

        The Cauchy distribution has no finite mean and no finite variance. The CLT
        requires both. Without them, the guarantee breaks down.
        """
    )
    st.latex(
        r"\text{Cauchy}(0,1): \quad f(x) = \frac{1}{\pi(1+x^2)}"
    )
    st.markdown(
        r"""
        The left panel shows the sum of $n$ independent Cauchy samples — the heavy
        tails never shrink. The right panel shows the sum of $n$ Normal samples for
        comparison, where the shape converges to a Gaussian as the CLT predicts
        (the spread grows as $\sqrt{n}$, but the shape becomes more bell-like).
        """
    )

    n_cauchy = st.slider("Number of terms summed (n)", 1, 200, 1, key="ncauchy")

    rng_c = np.random.default_rng(SEED)
    n_samples = 5000

    cauchy_sum  = stats.cauchy.rvs(size=(n_samples, n_cauchy),
                                   random_state=rng_c).sum(axis=1)
    normal_sum  = rng_c.standard_normal((n_samples, n_cauchy)).sum(axis=1)

    # clip Cauchy for display — tails are extreme
    clip = np.percentile(np.abs(cauchy_sum), 98)
    cauchy_clipped = cauchy_sum[np.abs(cauchy_sum) < clip]

    fig, axes = make_fig()

    axes[0].hist(cauchy_clipped, bins=80, density=True, color=C_POP, alpha=0.7)
    x_c = np.linspace(-clip, clip, 400)
    axes[0].plot(x_c, stats.cauchy.pdf(x_c / n_cauchy) / n_cauchy,
                 color=C_FIT, lw=2, label=f"Cauchy(0, {n_cauchy}) PDF")
    axes[0].set_title(f"Sum of {n_cauchy} Cauchy — still heavy-tailed")
    axes[0].set_xlabel("Value (extreme tails clipped for display)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    mu_n  = normal_sum.mean()
    sig_n = normal_sum.std()
    x_n   = np.linspace(mu_n - 4 * sig_n, mu_n + 4 * sig_n, 400)
    axes[1].hist(normal_sum, bins=60, density=True, color=C_HIST, alpha=0.7,
                 label=f"Sum of {n_cauchy} Normal(0,1)")
    axes[1].plot(x_n, stats.norm.pdf(x_n, mu_n, sig_n),
                 color=C_FIT, lw=2, label="Normal fit")
    axes[1].set_title(f"Sum of {n_cauchy} Normal — Gaussian shape as expected")
    axes[1].set_xlabel("Value")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Why doesn't the CLT apply here?"):
        st.markdown(
            r"""
            The CLT proof relies on the characteristic function (Fourier transform of
            the PDF) being expandable in a Taylor series around zero, which requires
            finite variance. The Cauchy characteristic function is
            $\phi(t) = e^{-|t|}$ — it has no Taylor expansion at $t=0$.

            Practically: the Cauchy distribution has such heavy tails that a single
            extreme outlier can dominate the sum of thousands of terms. The "average"
            never stabilises. In fact, the **average of $n$ Cauchy samples has exactly
            the same distribution as a single Cauchy sample** — averaging does nothing.

            This is why Cauchy appears in physics as the Lorentzian lineshape and in
            finance as a warning about fat-tailed risk models that assume Gaussian
            behaviour.
            """
        )

st.caption("© Joshua Osborne")
