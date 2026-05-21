import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("\n--- WEATHER DATA CLUSTERING STARTED ---\n")

# =========================================================
# CONFIGURATION
# =========================================================
FILE_PATH = "pakistan_aqi_weather.xlsx"
K = 5
# Colors for clusters 0-4 (blue, green, yellow, orange, red)
CLUSTER_COLORS = ["#2196F3", "#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]

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
# HELPER — CLUSTER + VISUALIZE ONE SHEET
# =========================================================
def process_sheet(df_raw: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    print(f"\n{'#'*60}")
    print(f"  Processing sheet: {sheet_name}")
    print(f"{'#'*60}")

    # ── separate time column ──────────────────────────────
    time_col = None
    if "time" in df_raw.columns:
        time_col = pd.to_datetime(df_raw["time"], errors="coerce")

    # ── numeric subset for clustering ────────────────────
    df_num = df_raw.select_dtypes(include=[np.number]).copy()
    df_num.dropna(inplace=True)
    valid_idx = df_num.index

    all_charts  = []
    cluster_cols = {}

    color_scale = alt.Scale(domain=list(range(K)), range=CLUSTER_COLORS)

    for obj_name, features in objectives.items():
        available = [f for f in features if f in df_num.columns]
        missing   = [f for f in features if f not in df_num.columns]
        if missing:
            print(f"  [{obj_name}] Missing features (skipped): {missing}")
        if len(available) < 2:
            print(f"  [{obj_name}] Not enough features — skipping.")
            continue

        print(f"\n  Objective : {obj_name}")
        print(f"  Features  : {available}")

        X  = df_num.loc[valid_idx, available].copy()
        Xs = StandardScaler().fit_transform(X)

        # ── KMeans ───────────────────────────────────────
        km     = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        X["cluster"] = labels

        # Only numeric cluster (0-4) written to Excel
        cluster_cols[f"cluster_{obj_name}"] = labels

        plot_df = X.copy().reset_index(drop=True)
        if time_col is not None:
            plot_df["time"] = time_col.loc[valid_idx].values

        # ── Chart 1: Elbow ───────────────────────────────
        inertias = []
        for k in range(1, 10):
            m = KMeans(n_clusters=k, random_state=42, n_init=10)
            m.fit(Xs)
            inertias.append(m.inertia_)
        elbow_df = pd.DataFrame({"k": range(1, 10), "inertia": inertias})

        elbow_line = (
            alt.Chart(elbow_df)
            .mark_line(point=True, color="#1976D2")
            .encode(
                x=alt.X("k:Q", title="Number of Clusters (k)",
                         axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("inertia:Q", title="Inertia"),
                tooltip=["k:Q", alt.Tooltip("inertia:Q", format=".1f")]
            )
        )
        elbow_rule = (
            alt.Chart(elbow_df[elbow_df["k"] == K])
            .mark_rule(color="red", strokeDash=[4, 4])
            .encode(x="k:Q")
        )
        elbow_chart = (elbow_line + elbow_rule).properties(
            title=f"{obj_name} — Elbow Method",
            width=280, height=200
        )

        # ── Chart 2: Scatter (first 2 features) ─────────
        scatter_df = pd.DataFrame({
            available[0]: Xs[:, 0],
            available[1]: Xs[:, 1],
            "cluster": plot_df["cluster"].astype(int)
        })
        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=20, opacity=0.55)
            .encode(
                x=alt.X(f"{available[0]}:Q", title=available[0]),
                y=alt.Y(f"{available[1]}:Q", title=available[1]),
                color=alt.Color(
                    "cluster:O",
                    scale=color_scale,
                    legend=alt.Legend(title="Cluster")
                ),
                tooltip=[alt.Tooltip("cluster:O", title="Cluster")]
            )
        ).properties(
            title=f"{obj_name} — Scatter: {available[0]} vs {available[1]}",
            width=280, height=200
        )

        # ── Chart 3: Bar — records per cluster ──────────
        bar_df = (
            plot_df.groupby("cluster")
            .size()
            .reset_index(name="count")
        )
        bar_df["cluster"] = bar_df["cluster"].astype(int)

        bar = (
            alt.Chart(bar_df)
            .mark_bar()
            .encode(
                x=alt.X("cluster:O", title="Cluster"),
                y=alt.Y("count:Q", title="Count"),
                color=alt.Color("cluster:O", scale=color_scale, legend=None),
                tooltip=["cluster:O", "count:Q"]
            )
        ) + (
            alt.Chart(bar_df)
            .mark_text(dy=-8, fontSize=11)
            .encode(
                x=alt.X("cluster:O"),
                y=alt.Y("count:Q"),
                text="count:Q"
            )
        )
        bar = bar.properties(
            title=f"{obj_name} — Records per Cluster",
            width=280, height=200
        )

        # ── Chart 4: Heatmap — feature means per cluster ─
        means = plot_df.groupby("cluster")[available].mean()
        norm  = (means - means.min()) / (means.max() - means.min() + 1e-9)
        heat_df = norm.reset_index().melt(
            id_vars="cluster", var_name="feature", value_name="value"
        )
        heat_df["cluster"] = heat_df["cluster"].astype(int)

        heatmap = (
            alt.Chart(heat_df)
            .mark_rect()
            .encode(
                x=alt.X("feature:N", title="Feature",
                         axis=alt.Axis(labelAngle=-35)),
                y=alt.Y("cluster:O", title="Cluster"),
                color=alt.Color(
                    "value:Q",
                    scale=alt.Scale(scheme="redyellowgreen"),
                    title="Norm. Mean"
                ),
                tooltip=["cluster:O", "feature:N",
                         alt.Tooltip("value:Q", format=".3f")]
            )
        ).properties(
            title=f"{obj_name} — Feature Means per Cluster (normalised)",
            width=400, height=180
        )

        # ── Chart 5: Clusters over time ──────────────────
        if time_col is not None:
            time_df = pd.DataFrame({
                "time":    plot_df["time"],
                "cluster": plot_df["cluster"].astype(int)
            })
            time_chart = (
                alt.Chart(time_df)
                .mark_circle(size=15, opacity=0.5)
                .encode(
                    x=alt.X("time:T", title="Time"),
                    y=alt.Y("cluster:O", title="Cluster"),
                    color=alt.Color("cluster:O", scale=color_scale,
                                    legend=alt.Legend(title="Cluster")),
                    tooltip=["time:T", "cluster:O"]
                )
            ).properties(
                title=f"{obj_name} — Clusters over Time",
                width=400, height=180
            )
        else:
            time_chart = None

        # ── Assemble objective block ──────────────────────
        row1 = (
            alt.hconcat(elbow_chart, scatter, bar)
            .resolve_scale(color="independent")
        )
        row2 = (
            alt.hconcat(heatmap, time_chart)
            .resolve_scale(color="independent")
        ) if time_chart else heatmap

        combined = (
            alt.vconcat(row1, row2)
            .resolve_scale(color="independent")
            .properties(title=alt.TitleParams(
                text=f"Objective: {obj_name}",
                fontSize=15, fontWeight="bold", anchor="start"
            ))
        )
        all_charts.append(combined)
        print(f"  Charts built for {obj_name}")

    # ── Save all charts to one HTML ───────────────────────
    if all_charts:
        final_chart = (
            alt.vconcat(*all_charts)
            .resolve_scale(color="independent")
        )
        html_path = f"viz_{sheet_name}.html"
        final_chart.save(html_path)
        print(f"\n  Visualisation saved → {html_path}")

    # ── Append only numeric cluster columns to original df ─
    df_out = df_raw.copy()
    for col_name, values in cluster_cols.items():
        s = pd.Series(values, index=valid_idx, name=col_name)
        df_out[col_name] = s

    return df_out

# =========================================================
# MAIN — READ ALL SHEETS, PROCESS EACH SEPARATELY
# =========================================================
xl     = pd.ExcelFile(FILE_PATH)
sheets = xl.sheet_names
print(f"Sheets found in file: {sheets}")

# Warn if 'Lahore' sheet is missing or name differs
if "Lahore" not in sheets:
    close = [s for s in sheets if s.strip().lower() == "lahore"]
    if close:
        print(f"  WARNING: 'Lahore' not found — closest match is '{close[0]}'")
    else:
        print(f"  WARNING: No 'Lahore' sheet found! Available: {sheets}")

output_files = []

for sheet in sheets:
    print(f"\nLoading sheet: '{sheet}'")
    df_sheet = xl.parse(sheet)
    print(f"  Shape   : {df_sheet.shape}")
    print(f"  Columns : {list(df_sheet.columns)}")

    df_result = process_sheet(df_sheet, sheet_name=sheet)

    out_path = f"weather_clustered_{sheet}.xlsx"
    df_result.to_excel(out_path, index=False)
    output_files.append(out_path)
    print(f"  Output saved → {out_path}")

# =========================================================
# SUMMARY
# =========================================================
print("\n" + "="*60)
print("  PROCESS COMPLETE")
print("  Output Excel files:")
for f in output_files:
    print(f"    - {f}")
print("  Visualisation HTML files:")
for s in sheets:
    print(f"    - viz_{s}.html")
print("\n  Cluster numbers: 0 = very low | 1 = Low | 2 = Moderate | 3 = High | 4 = Extreme")
print("="*60)