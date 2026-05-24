import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

print("\n--- WEATHER DATA CLUSTERING v3 STARTED ---\n")
print("  Logic order:")
print("  STEP 1 → Find outliers per feature, visualize each feature separately")
print("  STEP 2 → Select features per objective, cluster FULL data (outliers included)")
print("  STEP 3 → Visualize clusters — outliers placed in the cluster they fit best\n")

# =========================================================
# CONFIGURATION
# =========================================================
FILE_PATH        = "pakistan_aqi_weather.xlsx"
K                = 5        # number of clusters (0–4)
CONTAMINATION    = 0.0002   # Isolation Forest contamination per feature
IQR_MULTIPLIER   = 3.0      # extreme fence multiplier

CLUSTER_COLORS = ["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
CLUSTER_DOMAIN = list(range(K))   # [0, 1, 2, 3, 4]

objectives = {
    "Rainfall":     ['precipitation', 'rain', 'snowfall',
                    'relative_humidity_2m', 'cloud_cover',
                    'pressure_msl', 'dew_point_2m'],
    "Cloud":        ['cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
                    'cloud_cover_high', 'relative_humidity_2m',
                    'temperature_2m', 'dew_point_2m', 'aerosol_optical_depth'],
    "Solar":        ['shortwave_radiation', 'direct_radiation',
                    'diffuse_radiation', 'uv_index',
                    'uv_index_clear_sky', 'cloud_cover'],
    "HeatWave":     ['temperature_2m', 'apparent_temperature',
                    'relative_humidity_2m', 'dew_point_2m',
                    'uv_index', 'shortwave_radiation', 'wind_speed_10m'],
    "IndPollution": ['pm10', 'pm2_5', 'carbon_monoxide',
                    'nitrogen_dioxide', 'sulphur_dioxide',
                    'ozone', 'aerosol_optical_depth', 'dust'],
    "AQI":          ['pm10', 'pm2_5', 'carbon_monoxide',
                    'nitrogen_dioxide', 'sulphur_dioxide', 'ozone'],
    "AirPollution": ['pm10', 'pm2_5', 'nitrogen_dioxide',
                    'sulphur_dioxide', 'ozone', 'dust',
                    'wind_speed_10m', 'wind_direction_10m'],
}


# =========================================================
# STEP 1 — PER-FEATURE OUTLIER DETECTION
# =========================================================
def detect_feature_outliers(df_num: pd.DataFrame,
                            time_col: pd.Series | None,
                            contamination: float = CONTAMINATION,
                            iqr_multiplier: float = IQR_MULTIPLIER):
    feature_outlier_masks  = {}
    feature_outlier_charts = []

    numeric_features = df_num.columns.tolist()
    print(f"\n  [STEP 1] Detecting outliers for {len(numeric_features)} features individually...")

    for feat in numeric_features:
        col = df_num[[feat]].values   # shape (n, 1)

        # ── Gate 1: Isolation Forest ──────────────────────
        iso      = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        if_mask  = iso.fit_predict(col) == -1

        # ── Gate 2: IQR extreme fence ─────────────────────
        q1, q3   = np.percentile(col, [25, 75])
        iqr      = q3 - q1
        lo, hi   = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        iqr_mask = (col[:, 0] < lo) | (col[:, 0] > hi)

        # ── Dual gate intersection ────────────────────────
        mask     = if_mask & iqr_mask
        feature_outlier_masks[feat] = mask

        n_out = mask.sum()
        if n_out > 0:
            print(f"    {feat:40s} → {n_out:4d} outliers (IF:{if_mask.sum()} | IQR:{iqr_mask.sum()})")
        else:
            print(f"    {feat:40s} → no outliers")

        # ── Build per-feature charts ──────────────────────
        series      = df_num[feat].reset_index(drop=True)
        plot_series = pd.DataFrame({
            "value":      series,
            "is_outlier": mask
        })

        hist_base = alt.Chart(plot_series).mark_bar(opacity=0.6).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=50), title=feat),
            y=alt.Y("count()", title="Count"),
            color=alt.condition(alt.datum.is_outlier, alt.value("#F44336"), alt.value("#2196F3")),
            tooltip=["count()"]
        ).properties(width=340, height=180, title=f"{feat} — Distribution (red = outlier, n={n_out})")

        fence_df = pd.DataFrame({"x": [lo, hi]})
        fence    = alt.Chart(fence_df).mark_rule(color="#FF9800", strokeDash=[4, 3], strokeWidth=1.5).encode(x="x:Q")
        hist_chart = hist_base + fence

        if n_out > 0 and time_col is not None:
            ts_df = pd.DataFrame({
                "time":       time_col.values,
                "value":      series.values,
                "is_outlier": mask
            })
            ts_normal = (
                alt.Chart(ts_df[~ts_df["is_outlier"]])
                .mark_circle(size=6, opacity=0.25, color="#2196F3")
                .encode(x=alt.X("time:T", title="Time"), y=alt.Y("value:Q", title=feat), tooltip=["time:T", "value:Q"])
            )
            ts_outlier = (
                alt.Chart(ts_df[ts_df["is_outlier"]])
                .mark_circle(size=60, opacity=0.9, color="#F44336")
                .encode(x=alt.X("time:T"), y=alt.Y("value:Q"), tooltip=["time:T", "value:Q"])
            )
            ts_chart = (ts_normal + ts_outlier).properties(width=340, height=180, title=f"{feat} — Outliers over Time")
            combined = alt.hconcat(hist_chart, ts_chart).resolve_scale(color="independent")
        else:
            combined = hist_chart

        feature_outlier_charts.append(combined)

    return feature_outlier_masks, feature_outlier_charts


# =========================================================
# STEP 2 — BUILD OBJECTIVE-LEVEL OUTLIER SUMMARY TABLE
# =========================================================
def build_outlier_explanation(df_num: pd.DataFrame, feature_outlier_masks: dict, time_col: pd.Series | None) -> pd.DataFrame:
    records = []
    feat_stats = {feat: (df_num[feat].mean(), df_num[feat].std()) for feat in feature_outlier_masks}

    all_flagged = set()
    for feat, mask in feature_outlier_masks.items():
        all_flagged.update(np.where(mask)[0].tolist())

    for row_i in sorted(all_flagged):
        flagged_feats = [feat for feat, mask in feature_outlier_masks.items() if mask[row_i]]
        record = {
            "row_index":          row_i,
            "time":               time_col.iloc[row_i] if time_col is not None else None,
            "flagged_features":   ", ".join(flagged_feats),
            "n_features_flagged": len(flagged_feats),
        }
        for feat in flagged_feats:
            val          = df_num[feat].iloc[row_i]
            mu, sigma    = feat_stats[feat]
            zscore       = (val - mu) / (sigma + 1e-9)
            record[f"{feat}_value"]  = round(val, 4)
            record[f"{feat}_zscore"] = round(zscore, 2)

        n = len(flagged_feats)
        record["severity"] = "High" if n >= 3 else ("Medium" if n == 2 else "Low")
        records.append(record)

    return pd.DataFrame(records)


# =========================================================
# STEP 3 — CLUSTER FULL DATA
# =========================================================
def cluster_full_data(X: pd.DataFrame, clean_mask: np.ndarray, available: list) -> tuple:
    X_clean  = X[clean_mask]
    scaler   = StandardScaler()
    Xs_clean = scaler.fit_transform(X_clean[available])

    km       = KMeans(n_clusters=K, random_state=42, n_init=10)
    km.fit(Xs_clean)

    inertias = []
    for k in range(1, 10):
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        m.fit(Xs_clean)
        inertias.append(m.inertia_)

    Xs_full     = scaler.transform(X[available])
    labels_full = km.predict(Xs_full)

    return labels_full, scaler, km, inertias


# =========================================================
# MAIN SHEET PROCESSOR
# =========================================================
def process_sheet(df_raw: pd.DataFrame, sheet_name: str) -> tuple:
    print(f"\n{'#'*60}\n  Sheet: {sheet_name}\n{'#'*60}")

    time_col = None
    if "time" in df_raw.columns:
        time_col = pd.to_datetime(df_raw["time"], errors="coerce")

    df_num = df_raw.select_dtypes(include=[np.number]).copy().dropna()
    valid_idx = df_num.index
    df_num    = df_num.reset_index(drop=True)

    tc = time_col.loc[valid_idx].reset_index(drop=True) if time_col is not None else None

    # ── STEP 1: Outlier Identification ───────────────────
    feature_outlier_masks, feature_outlier_charts = detect_feature_outliers(df_num, tc)

    if feature_outlier_charts:
        rows = [alt.hconcat(*feature_outlier_charts[i:i+2]).resolve_scale(color="independent") 
                for i in range(0, len(feature_outlier_charts), 2)]
        feat_viz = alt.vconcat(*rows).properties(
            title=alt.TitleParams(text=f"STEP 1 — Per-Feature Outlier Detection ({sheet_name})", fontSize=16, anchor="start")
        )
        feat_viz.save(f"viz_{sheet_name}_feature_outliers.html")

    outlier_explanation_df = build_outlier_explanation(df_num, feature_outlier_masks, tc)
    print(f"\n  [STEP 1] {len(outlier_explanation_df)} records flagged as anomalies.")

    # ── STEP 2 & 3: Objective Mapping & Dynamic Plotting ─
    color_scale = alt.Scale(domain=CLUSTER_DOMAIN, range=CLUSTER_COLORS)
    all_obj_charts  = []
    cluster_cols    = {}

    for obj_name, features in objectives.items():
        available = [f for f in features if f in df_num.columns]
        if len(available) < 2: continue

        print(f"\n  [STEP 2] Clustering Objective: {obj_name} | Features: {available}")

        X = df_num[available].copy()
        obj_outlier_per_row = np.zeros(len(df_num), dtype=bool)
        for feat in available:
            if feat in feature_outlier_masks:
                obj_outlier_per_row |= feature_outlier_masks[feat]

        clean_mask = ~obj_outlier_per_row
        labels_full, scaler, km, inertias = cluster_full_data(df_num, clean_mask, available)
        cluster_cols[f"cluster_{obj_name}"] = labels_full

        # Plot Dataframe Preparation
        plot_df = X.copy()
        plot_df["cluster"] = labels_full
        plot_df["is_outlier"] = obj_outlier_per_row
        if tc is not None:
            plot_df["time"] = tc.values

        # Chart 1: Elbow method
        elbow_df = pd.DataFrame({"k": range(1, 10), "inertia": inertias})
        elbow_chart = (
            alt.Chart(elbow_df).mark_line(point=True, color="#1976D2").encode(
                x=alt.X("k:Q", title="Number of Clusters (k)", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("inertia:Q", title="Inertia"),
                tooltip=["k:Q", alt.Tooltip("inertia:Q", format=".1f")]
            ) + alt.Chart(elbow_df[elbow_df["k"] == K]).mark_rule(color="red", strokeDash=[4, 4]).encode(x="k:Q")
        ).properties(title=f"{obj_name} — Elbow Method", width=250, height=190)

        # Chart 2: Standardized Projection Scatter
        Xs_all = scaler.transform(X)
        scatter_df = pd.DataFrame({
            available[0]: Xs_all[:, 0], available[1]: Xs_all[:, 1],
            "cluster": labels_full.astype(int), "is_outlier": obj_outlier_per_row
        })
        scatter = alt.Chart(scatter_df).mark_point(opacity=0.6).encode(
            x=alt.X(f"{available[0]}:Q", title=available[0]),
            y=alt.Y(f"{available[1]}:Q", title=available[1]),
            color=alt.Color("cluster:O", scale=color_scale, legend=alt.Legend(title="Cluster")),
            shape=alt.Shape("is_outlier:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-up"]), legend=alt.Legend(title="Type")),
            size=alt.condition(alt.datum.is_outlier, alt.value(80), alt.value(25)),
            tooltip=["cluster:O", alt.Tooltip("is_outlier:N", title="Was outlier?")]
        ).properties(title=f"{obj_name} — Proj: {available[0]} vs {available[1]}", width=250, height=190)

        # Chart 3: Count per cluster distribution bar
        bar_df = plot_df.groupby("cluster").size().reset_index(name="count")
        bar = (
            alt.Chart(bar_df).mark_bar().encode(
                x=alt.X("cluster:O", title="Cluster"), y=alt.Y("count:Q", title="Count"),
                color=alt.Color("cluster:O", scale=color_scale, legend=None), tooltip=["cluster:O", "count:Q"]
            ) + alt.Chart(bar_df).mark_text(dy=-8, fontSize=11).encode(x="cluster:O", y="count:Q", text="count:Q")
        ).properties(title=f"{obj_name} — Records per Cluster", width=250, height=190)

        # Chart 4: Normalized Profile Heatmap
        means = plot_df.groupby("cluster")[available].mean()
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-9)
        heat_df = norm.reset_index().melt(id_vars="cluster", var_name="feature", value_name="value")
        heatmap = alt.Chart(heat_df).mark_rect().encode(
            x=alt.X("feature:N", axis=alt.Axis(labelAngle=-35), title="Feature"),
            y=alt.Y("cluster:O", title="Cluster"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redyellowgreen"), title="Norm. Mean"),
            tooltip=["cluster:O", "feature:N", alt.Tooltip("value:Q", format=".3f")]
        ).properties(title=f"{obj_name} — Centroid Matrix Profiles", width=380, height=175)

        # Chart 5: Historical Timeline Matrix Tracking
        if tc is not None:
            time_chart = alt.Chart(plot_df).mark_point(opacity=0.55).encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("cluster:O", title="Cluster"),
                color=alt.Color("cluster:O", scale=color_scale, legend=None),
                shape=alt.Shape("is_outlier:N", scale=alt.Scale(domain=[False, True], range=["circle", "triangle-up"]), legend=None),
                size=alt.condition(alt.datum.is_outlier, alt.value(60), alt.value(12)),
                tooltip=["time:T", "cluster:O", alt.Tooltip("is_outlier:N", title="Was Outlier?")]
            ).properties(title=f"{obj_name} — Chronological Clusters (▲=Anomalies)", width=380, height=175)
            row2 = alt.hconcat(heatmap, time_chart).resolve_scale(color="independent")
        else:
            row2 = heatmap

        row1 = alt.hconcat(elbow_chart, scatter, bar).resolve_scale(color="independent")
        combined = alt.vconcat(row1, row2).resolve_scale(color="independent").properties(
            title=alt.TitleParams(text=f"Objective Dashboard: {obj_name} (Outliers incorporated as Triangles)", fontSize=14, anchor="start")
        )
        all_obj_charts.append(combined)

    if all_obj_charts:
        alt.vconcat(*all_obj_charts).resolve_scale(color="independent").save(f"viz_{sheet_name}_clusters.html")

    df_out = df_raw.copy()
    for col_name, vals in cluster_cols.items():
        df_out[col_name] = pd.Series(vals, index=valid_idx)

    return df_out, outlier_explanation_df


# =========================================================
# CORE RUNTIME ENGINE Execution
# =========================================================
xl = pd.ExcelFile(FILE_PATH)
output_files = []

for sheet in xl.sheet_names:
    df_sheet = xl.parse(sheet)
    df_result, outlier_exp = process_sheet(df_sheet, sheet_name=sheet)

    out_path = f"weather_clustered_{sheet}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="Clustered_Data", index=False)
        if not outlier_exp.empty:
            outlier_exp.to_excel(writer, sheet_name="Outlier_Explanation", index=False)
    output_files.append(out_path)

print("\n" + "="*60 + "\n  PROCESS COMPLETE \n" + "="*60)