import streamlit as st
import pandas as pd
import numpy as np
from chart_modules.chart_quadrant import show_chart_quadrant  # ❌ REMOVE THIS LINE if this file is chart_quadrant.py


def generate_data(seed=42, n=400):
    np.random.seed(seed)
    missions = ['Surveillance', 'Training', 'Combat', 'Logistics']
    risks = list(range(5))
    mission = np.random.choice(missions, size=n)
    risk = np.random.choice(risks, size=n)
    breach_probs = [0.7 if m == 'Combat' and r == 4 else 0.6 if m == 'Logistics' and r == 2 else 0.4 if r >= 3 else 0.2
                    for m, r in zip(mission, risk)]
    breach = np.random.binomial(1, breach_probs)
    return pd.DataFrame({'Mission Type': mission, 'Cyber Risk Level': risk, 'Cyber Breach History': breach})


if "df" not in st.session_state or st.button("Regenerate Synthetic Data"):
    st.session_state.df = generate_data()

df = st.session_state.df.copy()

st.title("Air Force Cyber Breach Analysis Dashboard")

st.markdown("""
### Methods & Limitations
- This dashboard uses **Chi-Squared Tests** to evaluate whether observed differences in cyber breach rates across category intersections are statistically significant.
- Cells with **fewer than 10 total observations** are excluded from statistical testing to reduce the risk of false positives.
- For very small sample sizes, **Fisher’s Exact Test** would normally be more appropriate. However, Chi-Squared was chosen here due to the higher volume of data and speed of matrix-level testing.
- **The method used is shown in the chart tooltips and visual flags.** For example:
  - **Green triangle markers** represent statistically significant differences via **Chi-Squared**.
  - **Hover tooltips** display "Test: Chi-Squared" and the **exact p-value** for transparency.
  - (If Fisher’s Exact were used, the tooltip would state "Test: Fisher’s Exact".)
- Visual flags within the chart include:
  - **Chart legend labeled “Chi-Squared Significant”** near the top-right quadrant.
  - Tooltip format: _“12/3 breaches\nTest: Chi-Squared\np = 0.038”_
- All insights are auto-generated from synthetic data and dynamically adjust when new data is uploaded.
""")

st.markdown("This dashboard helps identify cyber breach patterns using rule-based stats, AI insights, and interactive visuals.")

st.markdown("### Optional Visual Deep Dive")
st.markdown("Use the toggle below to reveal an additional quadrant of AI-powered visualizations.")

if st.checkbox("Show Multi-Chart Visuals"):
    st.success("Quadrant visualizations loaded.")
    show_chart_quadrant(df)

    # Move chart interpretations inside the conditional so they only show when charts do
    st.markdown("### 📊 Scatter Plot Interpretation")
    st.markdown("""
    This chart reveals how mission type and cyber risk level jointly affect breach patterns.  
    Each cell's shade reflects the breach proportion, and Chi-Squared flags indicate statistically significant deviations.
    """)

    st.markdown("### 📉 Pareto Chart Interpretation")
    st.markdown("""
    This chart ranks mission-risk pairs by breach rate.  
    Bars with the highest breach likelihood appear first, highlighting priority areas for intervention.
    """)
