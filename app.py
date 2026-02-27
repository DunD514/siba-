import streamlit as st
import numpy as np
from scipy.stats import t
from scipy import stats

st.set_page_config(page_title="Two Sample t-Test", layout="centered")

st.title("Two Sample t-Test Calculator")

st.write("Enter values separated by commas (example: 10, 12, 14, 16)")

# Inputs
sample1_input = st.text_area("Sample 1")
sample2_input = st.text_area("Sample 2")

alternative = st.selectbox(
    "Alternative Hypothesis",
    ["two-sided", "left", "right"]
)

if st.button("Calculate"):

    try:
        # Convert input safely
        a = [float(x.strip()) for x in sample1_input.split(",") if x.strip() != ""]
        b = [float(x.strip()) for x in sample2_input.split(",") if x.strip() != ""]

        # Validate length
        if len(a) < 2 or len(b) < 2:
            st.error("Each sample must contain at least 2 numeric values.")
            st.stop()

        # Means
        mean1 = np.mean(a)
        mean2 = np.mean(b)

        # Standard deviations (sample SD)
        sd1 = np.std(a, ddof=1)
        sd2 = np.std(b, ddof=1)

        n1 = len(a)
        n2 = len(b)

        # Welch's t-test standard error
        se = np.sqrt((sd1**2)/n1 + (sd2**2)/n2)

        # If samples have no variability the standard error will be zero.
        if se == 0:
            st.error("Standard error is zero; t statistic is undefined. "
                     "This usually means one or both samples have zero variance.")
            st.stop()

        # t statistic
        t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite df
        df = ((sd1**2/n1 + sd2**2/n2)**2) / (
            ((sd1**2/n1)**2)/(n1-1) + ((sd2**2/n2)**2)/(n2-1)
        )

        # p-value calculation (manual)
        if alternative == "two-sided":
            p_value = 2 * (1 - t.cdf(abs(t_stat), df))
        elif alternative == "left":
            p_value = t.cdf(t_stat, df)
        else:  # right
            p_value = 1 - t.cdf(t_stat, df)

        # Display Results
        st.subheader("Results")
        st.write(f"Sample 1 Mean: {mean1:.4f}")
        st.write(f"Sample 2 Mean: {mean2:.4f}")
        st.write(f"T Statistic: {t_stat:.4f}")
        st.write(f"Degrees of Freedom (Welch): {df:.4f}")
        st.write(f"P-value: {p_value:.6f}")

        # SciPy verification (map the alternative terminology)
        st.subheader("SciPy Verification")
        alt_map = {"two-sided": "two-sided",
                   "left": "less",
                   "right": "greater"}
        scipy_test = stats.ttest_ind(a, b,
                                     alternative=alt_map[alternative],
                                     equal_var=False)
        st.write(scipy_test)

    except Exception as e:
        st.error(f"Error: {e}")