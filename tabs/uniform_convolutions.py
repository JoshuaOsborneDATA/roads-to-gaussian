import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import fftconvolve
from ._shared import C_HIST, C_FIT, make_fig


@st.cache_data
def _precompute(n_max=20):
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


def render():
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

    conv_results     = _precompute(n_max=20)
    x_conv, pdf_conv = conv_results[n_conv - 1]
    mu_n             = n_conv / 2
    sigma_n          = np.sqrt(n_conv / 12)
    x_std_conv       = (x_conv - mu_n) / sigma_n
    pdf_std_conv     = pdf_conv * sigma_n
    x_gauss          = np.linspace(-4, 4, 400)

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
