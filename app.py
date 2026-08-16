"""
Roads to the Gaussian
Interactive dashboard showing four ways the Normal distribution emerges.
"""

import streamlit as st

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
    st.info("Coming soon: Poisson to Gaussian convergence demo.")

with tab2:
    st.info("Coming soon: Uniform convolutions demo.")

with tab3:
    st.info("Coming soon: Log-normal and multiplicative CLT demo.")

with tab4:
    st.info("Coming soon: Bootstrap sampling distribution demo.")

st.caption("© Joshua Osborne")
