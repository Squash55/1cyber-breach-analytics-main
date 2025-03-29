import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def show_chart_quadrant(df):
    st.markdown("## 📊 Multi-Chart Risk Analytics Quadrant")
    col1, col2 = st.columns(2)

    bubble_df = df.groupby(['Mission Type', 'Cyber Risk Level']).agg(
        breach_rate=('Cyber Breach History', 'mean'),
        count=('Cyber Breach History', 'count')
    ).reset_index()

    with col1:
        st.markdown("#### 🔸 Radar Chart: Breach Risk by Mission Type")
        radar_df = df.groupby('Mission Type')['Cyber Breach History'].mean().reset_index()
        radar_df.columns = ['Mission Type', 'Breach Rate']
        radar_df = pd.concat([radar_df, radar_df.iloc[[0]]])
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df['Breach Rate'],
            theta=radar_df['Mission Type'],
            fill='toself'
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        st.markdown("#### 🔵 Bubble Chart: Breach Rate vs. Mission & Risk")
        fig_bubble = px.scatter(
            bubble_df,
            x='Mission Type',
            y='Cyber Risk Level',
            size='count',
            color='breach_rate',
            color_continuous_scale='RdBu',
            labels={'breach_rate': 'Breach %'}
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    with col1:
        st.markdown("#### 🌳 Decision Tree: Risk Guidance View")
        fig_tree, ax_tree = plt.subplots(figsize=(5, 4))
        ax_tree.text(0.1, 0.5, "Low Risk\n(<=2)", ha='center', va='center', bbox=dict(boxstyle="round", fc="lightblue"))
        ax_tree.text(0.1, 0.2, "⬅ Few Breaches", ha='center')
        ax_tree.text(0.5, 0.5, "Moderate Risk\n(Level 3)", ha='center', va='center', bbox=dict(boxstyle="round", fc="khaki"))
        ax_tree.text(0.5, 0.2, "↔ Mixed Results", ha='center')
        ax_tree.text(0.9, 0.5, "High Risk\n(Level 4)", ha='center', va='center', bbox=dict(boxstyle="round", fc="salmon"))
        ax_tree.text(0.9, 0.2, "➡ Mostly Breaches", ha='center')

        st.pyplot(fig_tree)

    with col2:
        st.markdown("#### 🔁 Sankey Diagram: Mission → Risk → Outcome")
        sankey_df = pd.DataFrame({
            'source': df['Mission Type'],
            'intermediate': df['Cyber Risk Level'].astype(str),
            'target': df['Cyber Breach History'].replace({0: 'No Breach', 1: 'Breach'})
        })

        link_1 = sankey_df.groupby(['source', 'intermediate']).size().reset_index(name='count')
        link_2 = sankey_df.groupby(['intermediate', 'target']).size().reset_index(name='count')

        labels = list(pd.unique(sankey_df[['source', 'intermediate', 'target']].values.ravel()))
        label_map = {label: i for i, label in enumerate(labels)}

        source_list = []
        target_list = []
        value_list = []

        for _, row in link_1.iterrows():
            source_list.append(label_map[row['source']])
            target_list.append(label_map[row['intermediate']])
            value_list.append(row['count'])

        for _, row in link_2.iterrows():
            source_list.append(label_map[row['intermediate']])
            target_list.append(label_map[row['target']])
            value_list.append(row['count'])

        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source_list,
                target=target_list,
                value=value_list
            )
        )])
        sankey_fig.update_layout(font_size=10)
        st.plotly_chart(sankey_fig, use_container_width=True)

    # === INTERPRETATIONS ===
    st.markdown("### 📈 Scatter Plot Interpretation")
    st.markdown("""
    This chart reveals how mission type and cyber risk level jointly affect breach patterns.  
    Each cell's shade reflects the breach proportion, and Chi-Squared flags indicate statistically significant deviations.
    """)

    st.markdown("### 📉 Pareto Chart Interpretation")
    st.markdown("""
    This chart ranks mission-risk pairs by breach rate.  
    Bars with the highest breach likelihood appear first, highlighting priority areas for intervention.
    """)

    # Placeholder: future chart interpretations (Radar, Sankey, Tree) could go here too
