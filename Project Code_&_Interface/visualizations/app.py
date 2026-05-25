import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# 1. APPLICATION CONSTANTS & CORE CONFIG
st.set_page_config(
    page_title="AQI Analytics, Clustering & Outlier Dashboard",
    page_icon="🍃",
    layout="wide"
)

FILE_PATH = os.path.join("dataset", "pakistan_aqi_weather.xlsx")
K = 5
CONTAMINATION = 0.0002   # Isolation Forest configuration
IQR_MULTIPLIER = 3.0     # Extreme outlier fence multiplier
CLUSTER_COLORS = ["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
CLUSTER_DOMAIN = list(range(K))

objectives = {
    "Rainfall":     ['precipitation', 'rain', 'snowfall', 'relative_humidity_2m', 'cloud_cover', 'pressure_msl', 'dew_point_2m'],
    "Cloud":        ['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'relative_humidity_2m', 'temperature_2m', 'dew_point_2m', 'aerosol_optical_depth'],
    "Solar":        ['shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'uv_index', 'uv_index_clear_sky', 'cloud_cover'],
    "HeatWave":     ['temperature_2m', 'apparent_temperature', 'relative_humidity_2m', 'dew_point_2m', 'uv_index', 'shortwave_radiation', 'wind_speed_10m'],
    "IndPollution": ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'aerosol_optical_depth', 'dust'],
    "AQI":          ['pm10', 'pm2_5', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'],
    "AirPollution": ['pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'ozone', 'dust', 'wind_speed_10m', 'wind_direction_10m'],
}

# 2. HELPER CACHED SYSTEM UTILITIES
@st.cache_data
def get_sheet_names():
    if not os.path.exists(FILE_PATH):
        return []
    return pd.ExcelFile(FILE_PATH).sheet_names

@st.cache_data
def load_sheet_data(sheet_name):
    return pd.read_excel(FILE_PATH, sheet_name=sheet_name)


# 3. GLOBAL SIDEBAR MAIN NAVIGATION INTERFACES
st.sidebar.title(" Navigation Menu")
app_mode = st.sidebar.radio(
    "Go to:", 
    ["📈 Forecasting Analysis", " Outlier Analysis Pipeline", "🧩 Weather & Pollution Clustering"]
)

sheets = get_sheet_names()
if not sheets and app_mode != "📈 Forecasting Analysis":
    st.error(f" Target context data file missing at: `{FILE_PATH}`. Please check directory layouts.")
    st.stop()


# =====================================================================
# MODE 1: FORECASTING ANALYSIS
# =====================================================================
if app_mode == "📈 Forecasting Analysis":
    st.title("📊 Air Quality Index (AQI) Model Performance Evaluation")
    
    st.markdown("---")

    # Karachi Forecast Display Card
    st.header(" Karachi City - PM2.5 Forecast Evaluation")
    col1_kar, col2_kar = st.columns([2, 1])
    with col1_kar:
        st.subheader("Visual Analysis")
        if os.path.exists("visualizations\karachi_chart.png"):
            st.image("visualizations\karachi_chart.png", caption="Karachi: 10-Month PM2.5 Test vs Predicted Profile", use_container_width=True)
        else:
            st.warning(" 'visualizationskarachi_chart.png' not found inside the workspace.")
    with col2_kar:
        st.subheader("📈 Model Metrics")
        st.metric(label="Model Accuracy", value="94.41%")
        st.markdown("**Performance Errors:**\n* **RMSE:** 2.69\n* **MAE:** 1.79\n* **MAPE:** 5.59%")
        st.info("💡 SARIMAX captures seasonality patterns over Karachi's weather transitions.")

    st.markdown("---")

    # Lahore Forecast Display Card
    st.header(" Lahore City - PM2.5 Forecast Evaluation")
    col1_lah, col2_lah = st.columns([2, 1])
    with col1_lah:
        st.subheader("Visual Analysis")
        if os.path.exists("visualizations\lahore_chart.png"):
            st.image("visualizations\lahore_chart.png", caption="Lahore: 10-Month PM2.5 Test vs Predicted Profile", use_container_width=True)
        else:
            st.warning(" 'lahore_chart.png' not found inside the workspace.")
    with col2_lah:
        st.subheader("📈 Model Metrics")
        st.metric(label="Model Accuracy", value="85.17%")
        st.markdown("**Performance Errors:**\n* **RMSE:** 13.35\n* **MAE:** 9.27\n* **MAPE:** 14.83%")
        st.info("💡 Model captures severe smog cycles distinct to Lahore's environment.")


# =====================================================================
# MODE 2: OUTLIER ANALYSIS PIPELINE (New Execution Matrix)
# =====================================================================
elif app_mode == " Outlier Analysis Pipeline":
    st.title(" Per-Feature Outlier Identification & Detection Pipeline")
    st.markdown("This execution engine uses a **Dual-Gate Intersection Filter** combining an `Isolation Forest` with an `IQR Extreme Fence` to filter system anomalies.")
    st.markdown("---")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Outlier Controls")
    selected_sheet = st.sidebar.selectbox("Select Target City/Sheet:", options=sheets)
    
    df_raw = load_sheet_data(selected_sheet)
    
    time_col = None
    if "time" in df_raw.columns:
        time_col = pd.to_datetime(df_raw["time"], errors="coerce")

    df_num = df_raw.select_dtypes(include=[np.number]).copy().dropna()
    valid_idx = df_num.index
    df_num = df_num.reset_index(drop=True)
    tc = time_col.loc[valid_idx].reset_index(drop=True) if time_col is not None else None

    # Step 1 Logic Execution
    feature_outlier_masks = {}
    numeric_features = df_num.columns.tolist()
    
    st.subheader(f"📊 Distribution & Temporal Charts for: {selected_sheet}")
    
    for feat in numeric_features:
        col_vals = df_num[[feat]].values
        
        # Isolation Forest Filter
        iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
        if_mask = iso.fit_predict(col_vals) == -1
        
        # IQR Fence limits
        q1, q3 = np.percentile(col_vals, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        iqr_mask = (col_vals[:, 0] < lo) | (col_vals[:, 0] > hi)
        
        # Intersect Gates
        mask = if_mask & iqr_mask
        feature_outlier_masks[feat] = mask
        n_out = mask.sum()
        
        # Plot Frame Assembly
        plot_series = pd.DataFrame({"value": df_num[feat], "is_outlier": mask})
        
        # Display features in expanding sections to avoid UI clutter
        with st.expander(f"📉 Feature Profile Analysis: {feat} ({n_out} Outliers Flagged)"):
            c1, c2 = st.columns(2)
            
            with c1:
                hist_base = alt.Chart(plot_series).mark_bar(opacity=0.65).encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title=feat),
                    y=alt.Y("count()", title="Frequency Count"),
                    color=alt.condition(alt.datum.is_outlier, alt.value("#F44336"), alt.value("#2196F3")),
                    tooltip=["count()"]
                )
                fence_df = pd.DataFrame({"x": [lo, hi]})
                fence = alt.Chart(fence_df).mark_rule(color="#FF9800", strokeDash=[4, 3], strokeWidth=1.5).encode(x="x:Q")
                st.altair_chart((hist_base + fence).properties(title=f"Distribution Profile Histogram (Orange Fences = 3.0*IQR)", height=230), use_container_width=True)
                
            with c2:
                if n_out > 0 and tc is not None:
                    ts_df = pd.DataFrame({"time": tc.values, "value": df_num[feat].values, "is_outlier": mask})
                    ts_normal = alt.Chart(ts_df[~ts_df["is_outlier"]]).mark_circle(size=8, opacity=0.3, color="#2196F3").encode(
                        x=alt.X("time:T", title="Timeline Date"), y=alt.Y("value:Q", title=feat), tooltip=["time:T", "value:Q"]
                    )
                    ts_outlier = alt.Chart(ts_df[ts_df["is_outlier"]]).mark_circle(size=65, opacity=0.9, color="#F44336").encode(
                        x=alt.X("time:T"), y=alt.Y("value:Q"), tooltip=["time:T", "value:Q"]
                    )
                    st.altair_chart((ts_normal + ts_outlier).properties(title=f"Chronological Outliers Over Time Trace (Red Alerts)", height=230), use_container_width=True)
                else:
                    st.text_area(
                        "Chronological Logging Status", 
                        value="No structural or temporal timeline anomalies recognized inside this feature index boundary.", 
                        height=100,
                        key=f"txt_area_{feat}"  # <-- This key fixes the duplicate ID issue
                    )
    # Step 2 Log Table Construction
    st.markdown("---")
    st.subheader("📝 Explanatory Anomaly Severity Log Ledger")
    
    feat_stats = {feat: (df_num[feat].mean(), df_num[feat].std()) for feat in feature_outlier_masks}
    all_flagged = set()
    for feat, mask in feature_outlier_masks.items():
        all_flagged.update(np.where(mask)[0].tolist())
        
    records = []
    for row_i in sorted(all_flagged):
        flagged_feats = [feat for feat, mask in feature_outlier_masks.items() if mask[row_i]]
        record = {
            "Row Index Location": row_i,
            "Timestamp context": tc.iloc[row_i] if tc is not None else None,
            "Total Flags": len(flagged_feats),
            "Flagged Features": ", ".join(flagged_feats),
        }
        for feat in flagged_feats:
            val = df_num[feat].iloc[row_i]
            mu, sigma = feat_stats[feat]
            zscore = (val - mu) / (sigma + 1e-9)
            record[f"{feat} (Observed)"] = round(val, 2)
            record[f"{feat} (Z-Score)"] = round(zscore, 2)
            
        n = len(flagged_feats)
        record["Severity Rank"] = "🚨 High" if n >= 3 else (" Medium" if n == 2 else "ℹ️ Low")
        records.append(record)
        
    if records:
        log_df = pd.DataFrame(records)
        # Reorder columns to place key info first
        col_order = ["Row Index Location", "Timestamp context", "Severity Rank", "Total Flags", "Flagged Features"]
        remaining_cols = [c for c in log_df.columns if c not in col_order]
        st.dataframe(log_df[col_order + remaining_cols], use_container_width=True)
    else:
        st.success("🎉 Excellent! Zero collective dataset vector anomalies were uncovered based on configured evaluation thresholds.")


# =====================================================================
# MODE 3: WEATHER & POLLUTION CLUSTERING (With Integrated Outliers)
# =====================================================================
else:
    st.title("🧩 Weather & Data Clustering Insights (K-Means)")
    st.markdown("Select a sheet and matching objective. **Note**: Outliers are evaluated, but preserved inside plots using a dynamic tracking **Triangle shape (▲)** marker.")
    st.markdown("`Cluster Reference: 0 = Very Low | 1 = Low | 2 = Moderate | 3 = High | 4 = Extreme`")
    st.markdown("---")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Clustering Adjustments")
    selected_sheet = st.sidebar.selectbox("Select Target City/Sheet:", options=sheets)
    selected_obj = st.sidebar.selectbox("Select Cluster Objective:", options=list(objectives.keys()))

    df_raw = load_sheet_data(selected_sheet)
    
    time_col = None
    if "time" in df_raw.columns:
        time_col = pd.to_datetime(df_raw["time"], errors="coerce")

    df_num = df_raw.select_dtypes(include=[np.number]).copy().dropna()
    valid_idx = df_num.index
    df_num = df_num.reset_index(drop=True)
    tc = time_col.loc[valid_idx].reset_index(drop=True) if time_col is not None else None

    features = objectives[selected_obj]
    available = [f for f in features if f in df_num.columns]
    missing = [f for f in features if f not in df_num.columns]

    if missing:
        st.sidebar.warning(f"💡 Missing features dropped from sheet: {missing}")
        
    if len(available) < 2:
        st.error(f" Target dimension counts insufficient inside '{selected_sheet}' for matching the '{selected_obj}' schema context.")
    else:
        # Pre-calculate Outliers for filtering training cluster targets
        feature_outlier_masks = {}
        obj_outlier_per_row = np.zeros(len(df_num), dtype=bool)
        
        for feat in available:
            col_vals = df_num[[feat]].values
            iso = IsolationForest(contamination=CONTAMINATION, random_state=42, n_jobs=-1)
            if_mask = iso.fit_predict(col_vals) == -1
            q1, q3 = np.percentile(col_vals, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            iqr_mask = (col_vals[:, 0] < lo) | (col_vals[:, 0] > hi)
            
            mask = if_mask & iqr_mask
            obj_outlier_per_row |= mask

        clean_mask = ~obj_outlier_per_row
        
        # Fit models on Clean Subset, Predict on Full Dataset
        X = df_num[available].copy()
        X_clean = df_num[clean_mask]
        
        scaler = StandardScaler()
        Xs_clean = scaler.fit_transform(X_clean[available])
        
        km = KMeans(n_clusters=K, random_state=42, n_init=10)
        km.fit(Xs_clean)
        
        Xs_full = scaler.transform(X[available])
        labels_full = km.predict(Xs_full)

        # Plot frame setup
        plot_df = X.copy()
        plot_df["cluster"] = labels_full
        plot_df["is_outlier"] = obj_outlier_per_row
        if tc is not None:
            plot_df["time"] = tc.values

        color_scale = alt.Scale(domain=CLUSTER_DOMAIN, range=CLUSTER_COLORS)

        st.subheader(f"📊 Displaying Cluster Visualizations for: {selected_sheet} ({selected_obj})")
        
        # --- ROW 1: ELBOW CHART & SCATTER PLOT (With Triangles) ---
        row1_col1, row1_col2 = st.columns(2)

        with row1_col1:
            inertias = []
            for k in range(1, 10):
                m = KMeans(n_clusters=k, random_state=42, n_init=10)
                m.fit(Xs_clean)
                inertias.append(m.inertia_)
            elbow_df = pd.DataFrame({"k": range(1, 10), "inertia": inertias})

            elbow_line = alt.Chart(elbow_df).mark_line(point=True, color="#1976D2").encode(
                x=alt.X("k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("inertia:Q", title="Inertia"),
                tooltip=["k:Q", alt.Tooltip("inertia:Q", format=".1f")]
            )
            elbow_rule = alt.Chart(elbow_df[elbow_df["k"] == K]).mark_rule(color="red", strokeDash=[4, 4]).encode(x="k:Q")
            st.altair_chart((elbow_line + elbow_rule).properties(title="Optimal K Profile (Evaluated on Clean Data)", height=250), use_container_width=True)

        with row1_col2:
            scatter_df = pd.DataFrame({
                available[0]: Xs_full[:, 0],
                available[1]: Xs_full[:, 1],
                "cluster": plot_df["cluster"].astype(int),
                "is_outlier": obj_outlier_per_row
            })
            scatter = alt.Chart(scatter_df).mark_point(opacity=0.65).encode(
                x=alt.X(f"{available[0]}:Q", title=f"Scaled {available[0]}"),
                y=alt.Y(f"{available[1]}:Q", title=f"Scaled {available[1]}"),
                color=alt.Color("cluster:O", scale=color_scale, legend=alt.Legend(title="Cluster")),
                shape=alt.Shape("is_outlier:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-up"]), legend=alt.Legend(title="Anomaly Alert")),
                size=alt.condition(alt.datum.is_outlier, alt.value(85), alt.value(25)),
                tooltip=["cluster:O", alt.Tooltip("is_outlier:N", title="Outlier Record Flag?")]
            ).properties(title=f"Feature Dimensions Spatial Space (▲ = Outlier Elements)", height=250).interactive()
            st.altair_chart(scatter, use_container_width=True)

        # --- ROW 2: BAR CHART & HEATMAP PROFILE ---
        st.markdown("---")
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            bar_df = plot_df.groupby("cluster").size().reset_index(name="count")
            bar_df["cluster"] = bar_df["cluster"].astype(int)
            bar = alt.Chart(bar_df).mark_bar().encode(
                x=alt.X("cluster:O", title="Cluster Group ID"),
                y=alt.Y("count:Q", title="Data Allocation Counts"),
                color=alt.Color("cluster:O", scale=color_scale, legend=None),
                tooltip=["cluster:O", "count:Q"]
            )
            bar_text = alt.Chart(bar_df).mark_text(dy=-8, fontSize=11).encode(
                x=alt.X("cluster:O"), y=alt.Y("count:Q"), text="count:Q"
            )
            st.altair_chart((bar + bar_text).properties(title="Total Data Log Allocations Per Cluster", height=250), use_container_width=True)

        with row2_col2:
            means = plot_df.groupby("cluster")[available].mean()
            norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
            heat_df = norm.reset_index().melt(id_vars="cluster", var_name="feature", value_name="value")
            heat_df["cluster"] = heat_df["cluster"].astype(int)

            heatmap = alt.Chart(heat_df).mark_rect().encode(
                x=alt.X("feature:N", title="Evaluated Features Profile Metric", axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("cluster:O", title="Cluster Group ID"),
                color=alt.Color("value:Q", scale=alt.Scale(scheme="redyellowgreen"), title="Normalized Value"),
                tooltip=["cluster:O", "feature:N", alt.Tooltip("value:Q", format=".3f")]
            ).properties(title="Cluster Centroid Attribute Density Matrix Profiles", height=250)
            st.altair_chart(heatmap, use_container_width=True)

        # --- ROW 3: TIMELINE TRACKING ---
        if tc is not None:
            st.markdown("---")
            st.subheader("⏳ Temporal Cluster Distribution Matrix Timeline")
            time_chart = alt.Chart(plot_df).mark_point(opacity=0.55).encode(
                x=alt.X("time:T", title="Timeline Date Registration Log"),
                y=alt.Y("cluster:O", title="Cluster Group Placement State"),
                color=alt.Color("cluster:O", scale=color_scale, legend=None),
                shape=alt.Shape("is_outlier:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-up"]), legend=None),
                size=alt.condition(alt.datum.is_outlier, alt.value(70), alt.value(15)),
                tooltip=["time:T", "cluster:O", alt.Tooltip("is_outlier:N", title="Anomalous Row?")]
            ).properties(height=210).interactive()
            st.altair_chart(time_chart, use_container_width=True)

        # --- DATA EXPORTER OVERVIEW EXPANDER ---
        st.markdown("---")
        with st.expander("📝 View Processed Data Frame Sheet & Cluster Target Mapping"):
            df_out = df_raw.copy()
            df_out[f"cluster_{selected_obj}"] = pd.Series(labels_full, index=valid_idx)
            df_out[f"is_outlier_{selected_obj}"] = pd.Series(obj_outlier_per_row, index=valid_idx)
            st.dataframe(df_out, use_container_width=True)