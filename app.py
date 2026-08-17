"""
Roads to the Gaussian
Interactive dashboard showing four ways the Normal distribution emerges,
and one famous exception that doesn't.
"""

import streamlit as st
from tabs import poisson, uniform_convolutions, lognormal, bootstrap, random_walk, cauchy

st.set_page_config(
    page_title="Roads to the Gaussian",
    page_icon="🔔",
    layout="wide",
)

st.title("Roads to the Gaussian")

st.markdown(
    r"""
    ### The Central Limit Theorem

    The **Central Limit Theorem (CLT)** is one of the most important results in
    all of probability theory. In plain terms: if you add together a large number
    of independent random variables that each have a finite mean and variance, the
    sum will be approximately Normally distributed — regardless of what the
    individual variables look like.

    This is not a niche result. The CLT is the reason confidence intervals work,
    why hypothesis tests are valid, why bootstrap resampling is reliable, and why
    so much of classical statistics holds up in practice. It shows up in physics,
    finance, biology, engineering, and data science so routinely that it can start
    to feel like background noise — something assumed rather than noticed.

    This dashboard is an attempt to make it noticeable again. Each tab demonstrates
    a different path to the Gaussian: a different generative mechanism, a different
    starting distribution, a different framing. The last tab shows what happens when
    the CLT conditions are violated — a distribution that never converges, no matter
    how many terms you add.
    """
)

st.markdown("---")
st.markdown(
    """
    **What each tab shows:**

    | Tab | Starting point | Mechanism |
    |-----|---------------|-----------|
    | Poisson → Gaussian | Discrete count distribution | Poisson additivity: sum of λ independent Poisson(1) terms; CLT applies as λ grows |
    | Uniform Convolutions → Gaussian | Flat distribution | Repeated convolution smooths any shape toward a bell curve |
    | Log-normal → Gaussian | Skewed multiplicative process | Logarithm converts products to sums; CLT applies in log-space |
    | Bootstrap → Gaussian | Arbitrary (trimodal) population | Sample means are averages; CLT guarantees their distribution |
    | Random Walk → Gaussian | Two point masses (±1) | Cumulative steps are a running sum; shape converges with time |
    | The Exception: Cauchy | Heavy-tailed distribution | No finite variance — the CLT condition fails, convergence never happens |
    """
)

st.markdown("---")
st.caption(
    "Made by Joshua Osborne in collaboration with [Claude Code](https://claude.ai/code)."
)
st.markdown("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Poisson → Gaussian",
    "Uniform Convolutions → Gaussian",
    "Log-normal → Gaussian",
    "Bootstrap → Gaussian",
    "Random Walk → Gaussian",
    "The Exception: Cauchy",
])

with tab1:
    poisson.render()

with tab2:
    uniform_convolutions.render()

with tab3:
    lognormal.render()

with tab4:
    bootstrap.render()

with tab5:
    random_walk.render()

with tab6:
    cauchy.render()

st.caption("© Joshua Osborne · Made in collaboration with Claude Code")
