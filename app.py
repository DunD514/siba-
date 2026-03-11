import streamlit as st
import numpy as np
from scipy.stats import t
from statistics import stdev

st.title("Two Sample T-Test Calculator")
st.write("Enter sample values separated by commas and choose the type of hypothesis test.")

# Inputs
sample1 = st.text_input("Enter Sample 1 values (comma separated)")
sample2 = st.text_input("Enter Sample 2 values (comma separated)")

alternative = st.selectbox(
    "Select Alternative Hypothesis",
    ["two-tailed", "right-tailed", "left-tailed"]
)

def ttest(a, b, alternative):
    n1 = len(a)
    n2 = len(b)

    std1 = stdev(a)
    std2 = stdev(b)

    xbar1 = np.mean(a)
    xbar2 = np.mean(b)

    se = np.sqrt((std1**2/n1) + (std2**2/n2))
    tcalc = (xbar1 - xbar2) / se

    alpha = 0.05
    df = n1 + n2 - 2

    st.subheader("Results")
    st.write("Degrees of Freedom:", df)
    st.write("Calculated t-value:", round(tcalc,4))

    if alternative == "two-tailed":
        alpha = alpha / 2
        tpositive = t.ppf(1 - alpha, df)
        tnegative = t.ppf(alpha, df)

        pvalue = 2 * (1 - t.cdf(abs(tcalc), df))

        st.write("Critical t (+):", round(tpositive,4))
        st.write("Critical t (-):", round(tnegative,4))
        st.write("p-value:", round(pvalue,6))

    elif alternative == "right-tailed":
        tpositive = t.ppf(1 - alpha, df)
        pvalue = (1 - t.cdf(tcalc, df))

        st.write("Critical t:", round(tpositive,4))
        st.write("p-value:", round(pvalue,6))

    else:
        tnegative = t.ppf(alpha, df)
        pvalue = t.cdf(tcalc, df)

        st.write("Critical t:", round(tnegative,4))
        st.write("p-value:", round(pvalue,6))


# Button to run test
if st.button("Run T-Test"):
    try:
        a = list(map(float, sample1.split(",")))
        b = list(map(float, sample2.split(",")))

        ttest(a, b, alternative)

    except:
        st.error("Please enter valid numbers separated by commas.")
