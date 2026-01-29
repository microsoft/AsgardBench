"""
Live plan generation monitor - displays plan stats and updates in real-time.

Usage:
    streamlit run Magmathor/Utils/plan_monitor.py

The monitor will refresh every 2 seconds to show the latest plan generation stats.
"""

import time

import streamlit as st

from Magmathor.Utils.count_plans import count_plans, get_current_task, get_family_name


def main():
    st.set_page_config(
        page_title="Plan Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Plan Generation Monitor")

    # Show current task being worked on
    current_task = get_current_task()
    if current_task:
        st.info(f"🔧 **Currently working on:** {current_task}")

    # Settings in sidebar
    with st.sidebar:
        st.header("Settings")
        refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 2)
        max_samples = st.number_input("Max samples threshold", 1, 20, 3)
        auto_refresh = st.checkbox("Auto-refresh", value=True)

        if st.button("🔄 Refresh Now"):
            st.rerun()

    # Load plan stats
    plan_stats = count_plans()

    if not plan_stats.results_dict:
        st.info("No plans found. Start the plan generator to see stats here.")
    else:
        # Summary metrics at the top
        totals = {
            "success": 0,
            "success_inj": 0,
            "failures": 0,
            "existing": 0,
            "existing_inj": 0,
        }

        for task, results in plan_stats.results_dict.items():
            totals["success"] += len(results.success)
            totals["success_inj"] += len(results.success_inj)
            totals["failures"] += len(results.failures)
            totals["existing"] += len(results.existing)
            totals["existing_inj"] += len(results.existing_inj)

        total_normal = totals["success"] + totals["existing"]
        total_inj = totals["success_inj"] + totals["existing_inj"]
        total_all = total_normal + total_inj

        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Plans", f"{total_all}", f"{total_normal}+{total_inj}")
        with col2:
            st.metric(
                "Success",
                f"{totals['success'] + totals['success_inj']}",
                f"{totals['success']}+{totals['success_inj']}",
            )
        with col3:
            st.metric("Failures", totals["failures"])
        with col4:
            st.metric(
                "Existing",
                f"{totals['existing'] + totals['existing_inj']}",
                f"{totals['existing']}+{totals['existing_inj']}",
            )
        with col5:
            st.metric("Total Steps", plan_stats.total_steps)

        st.divider()

        # Build table data
        table_data = []
        for task in sorted(plan_stats.results_dict.keys()):
            results = plan_stats.results_dict[task]

            num_success = len(results.success)
            num_success_inj = len(results.success_inj)
            num_failures = len(results.failures)
            num_existing = len(results.existing)
            num_existing_inj = len(results.existing_inj)

            total_normal = num_success + num_existing
            total_inj = num_success_inj + num_existing_inj

            # Get scenes
            scenes = ""
            if task in plan_stats.scene_dict and plan_stats.scene_dict[task]:
                scene_numbers = sorted(plan_stats.scene_dict[task], key=int)
                scenes = ", ".join(scene_numbers)

            # Get step stats
            steps = ""
            if results.min_steps > 0 or results.max_steps > 0:
                steps = (
                    f"{results.min_steps}-{results.max_steps} ({results.avg_steps:.1f})"
                )

            # Determine if task is "full" (reached max_samples)
            family_name = get_family_name(task)
            if family_name is not None:
                # For family-based tasks (distribute__), check family total against 3x max
                family_total = plan_stats.num_family_plans(task)
                is_full = family_total >= max_samples * 3
            else:
                # For regular tasks, check individual total
                is_full = total_normal >= max_samples

            table_data.append(
                {
                    "Task": task,
                    "Total": f"{total_normal}+{total_inj}",
                    "Success": f"{num_success}+{num_success_inj}",
                    "Failures": num_failures,
                    "Existing": f"{num_existing}+{num_existing_inj}",
                    "Steps": steps,
                    "Scenes": scenes,
                    "_success": num_success,
                    "_existing": num_existing,
                    "_failures": num_failures,
                    "_is_full": is_full,
                }
            )

        # Display as styled dataframe
        if table_data:
            import pandas as pd

            df = pd.DataFrame(table_data)

            # Style function for rows
            def style_row(row):
                styles = []
                for col in row.index:
                    if col.startswith("_"):
                        styles.append("display: none;")
                    elif row["_failures"] > 0:
                        styles.append("background-color: #4d1a1a; color: #ff6b6b;")
                    elif row["_is_full"]:
                        styles.append("background-color: #1a4d1a; color: #4dff4d;")
                    elif row["_success"] > 0:
                        styles.append("background-color: #1a3d1a; color: #6bff6b;")
                    else:
                        styles.append("")
                return styles

            # Hide internal columns and apply styling
            display_cols = [
                "Task",
                "Total",
                "Success",
                "Failures",
                "Existing",
                "Steps",
                "Scenes",
            ]
            display_df = df[display_cols]

            def style_display_row(row):
                # Get the original row data for coloring logic
                orig_row = df.loc[row.name]
                return style_row(orig_row)[: len(display_cols)]

            styled_df = display_df.style.apply(style_display_row, axis=1)

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                height=600,
            )

        # Show last update time
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
