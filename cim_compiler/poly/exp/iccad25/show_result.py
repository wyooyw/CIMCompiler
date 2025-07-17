from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from db import DataBase

# Set page config
st.set_page_config(layout="wide")


# Connect to database
# @st.cache_resource
def get_database():
    return DataBase("iccad25.db")


db = get_database()

# Sidebar - Experiment List
st.sidebar.title("实验列表")

# Get all experiments
experiments = db.get_all_experiments()
selected_exp = None

for exp in experiments:
    # Format timestamp to readable date
    date_str = datetime.fromtimestamp(exp.time // 1000).strftime("%Y-%m-%d %H:%M:%S")

    # Create a container for each experiment
    with st.sidebar.container():
        # Display experiment info in a box with gray background
        st.markdown(
            f"""
            <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
            <b>实验 #{exp.id}</b><br>
            创建时间: {date_str}<br>
            描述: {exp.message}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        # View details button in second column
        if col2.button("查看详情", key=f"view_{exp.id}", use_container_width=True):
            selected_exp = exp

        # Add some spacing between experiments
        st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Main content area
if selected_exp:
    st.title(f"实验 #{selected_exp.id} 详情")

    # Get results for selected experiment
    results = db.get_experiment_results_by_experiment_id(selected_exp.id)

    results = sorted(results, key=lambda x: x.op)

    # Convert results to DataFrame
    df = pd.DataFrame([vars(r) for r in results])

    # Section 1: Success Statistics
    st.header("执行状态统计")

    success_count = len(df[df["status"] == 3])
    total_count = len(df)
    st.write(f"成功数 / 总数: {success_count} / {total_count}")

    # Show failed results if any
    failed_df = df[df["status"] != 3]
    if not failed_df.empty:
        with st.expander("查看失败记录"):
            st.dataframe(failed_df)

    # Section 4: Utilization Chart
    st.header("利用率统计")

    # Prepare data for utilization plotting
    util_plot_df = df.copy()

    fig_util = px.bar(
        util_plot_df,
        x="op",
        y="macro_ultilization",
        color="strategy",
        barmode="group",
        title="Operation Macro Utilization by Strategy",
        labels={"macro_ultilization": "Utilization Rate (%)"},
    )

    # Set y-axis range from 0 to 100
    fig_util.update_layout(yaxis_range=[0, 100])

    st.plotly_chart(fig_util, use_container_width=True)

    # Section 2: Latency Chart
    st.header("延迟统计")

    # Add toggle switch for absolute/relative values
    show_relative = (
        True  # st.checkbox("显示加速比（以im2col为基准）", key="latency_relative")
    )

    # Prepare data for plotting
    plot_df = df.copy()
    if show_relative:
        # Calculate speedup compared to im2col
        baseline = df[df["strategy"] == "im2col"].groupby("op")["latency"].first()
        plot_df = df.copy()
        plot_df["latency"] = plot_df.apply(
            lambda x: baseline[x["op"]]
            / x["latency"],  # Changed division order for speedup
            axis=1,
        )
        y_title = "Speedup (im2col=1)"
    else:
        y_title = "Latency"

    fig_latency = px.bar(
        plot_df,
        x="op",
        y="latency",
        color="strategy",
        barmode="group",
        title="Operation Latency by Strategy",
        labels={"latency": y_title},
    )
    st.plotly_chart(fig_latency, use_container_width=True)

    # Section 3: Energy Chart
    st.header("能耗统计")

    # Prepare data for energy plotting
    energy_plot_df = df.copy()
    # Calculate relative energy compared to im2col (im2col = 1)
    energy_baseline = df[df["strategy"] == "im2col"].groupby("op")["energy"].first()
    energy_plot_df["energy"] = energy_plot_df.apply(
        lambda x: x["energy"] / energy_baseline[x["op"]], axis=1
    )

    fig_energy = px.bar(
        energy_plot_df,
        x="op",
        y="energy",
        color="strategy",
        barmode="group",
        title="Operation Energy by Strategy",
        labels={"energy": "Relative Energy (im2col=1)"},
    )
    st.plotly_chart(fig_energy, use_container_width=True)
