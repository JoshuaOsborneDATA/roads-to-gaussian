import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ._shared import SEED, C_HIST, C_FIT, C_POP, make_fig


def render():
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

    rng       = np.random.default_rng(SEED)
    n_samples = 5000

    cauchy_sum = stats.cauchy.rvs(size=(n_samples, n_cauchy),
                                  random_state=rng).sum(axis=1)
    normal_sum = rng.standard_normal((n_samples, n_cauchy)).sum(axis=1)

    clip           = np.percentile(np.abs(cauchy_sum), 98)
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
