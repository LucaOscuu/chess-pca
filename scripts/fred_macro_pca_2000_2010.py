import io
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# 1. CONFIGURATION
# =========================

START_DATE = "2000-01-01"
END_DATE   = "2010-12-31"

# Suggested FRED series
SERIES = {
    "VIXCLS": "VIX",
    "BAA10Y": "BAA_10Y_Spread",
    "TEDRATE": "TED_Spread",
    "SP500": "SP500",
    "CSUSHPINSA": "CaseShiller_HPI",
    "MDSP": "Mortgage_Debt_Service",
    "TOTBKCR": "Total_Bank_Credit",
    "TOTALSL": "Total_Consumer_Credit",
    "INDPRO": "Industrial_Production",
    "UNRATE": "Unemployment_Rate",
    "T10Y2Y": "Yield_Curve_10Y_2Y",
    "NFCI": "Chicago_Financial_Conditions"
}

# Which variables to transform into percentage changes / returns
# This is optional, but usually better than using raw levels for trending series.
PCT_CHANGE_COLS = [
    "SP500",
    "CSUSHPINSA",
    "TOTBKCR",
    "TOTALSL",
    "INDPRO"
]

# Which columns to leave in levels
LEVEL_COLS = [
    "VIXCLS",
    "BAA10Y",
    "TEDRATE",
    "MDSP",
    "UNRATE",
    "T10Y2Y",
    "NFCI"
]

# =========================
# 2. DOWNLOAD FROM FRED
# =========================

def download_fred_series(series_id: str) -> pd.DataFrame:
    """
    Downloads one series from FRED using the CSV endpoint.
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df

dfs = []
for sid in SERIES.keys():
    print(f"Downloading {sid} ...")
    dfs.append(download_fred_series(sid))

# Merge all series on DATE
data = dfs[0]
for df in dfs[1:]:
    data = data.merge(df, on="DATE", how="outer")

# Filter date range
data = data[(data["DATE"] >= START_DATE) & (data["DATE"] <= END_DATE)].copy()
data = data.sort_values("DATE").reset_index(drop=True)

# =========================
# 3. MONTHLY RESAMPLING
# =========================

# Set date index
data = data.set_index("DATE")

# Convert all to monthly frequency.
# Using monthly mean is acceptable for a first pass.
monthly = data.resample("M").mean()

# =========================
# 4. TRANSFORMATIONS
# =========================

transformed = monthly.copy()

# Percentage change for trending level variables
for col in PCT_CHANGE_COLS:
    if col in transformed.columns:
        transformed[col] = transformed[col].pct_change() * 100.0

# Keep the rest in levels
# Already done implicitly for LEVEL_COLS

# Drop initial NA created by pct_change
transformed = transformed.dropna(how="all")

# Interpolate / forward fill small gaps
transformed = transformed.interpolate(method="time").ffill().bfill()

# Final drop of any remaining NA
transformed = transformed.dropna()

# Rename columns to readable labels
transformed = transformed.rename(columns=SERIES)

# =========================
# 5. STANDARDIZATION + PCA
# =========================

X = transformed.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    Z,
    index=X.index,
    columns=["PC1", "PC2"]
)

explained = pca.explained_variance_ratio_
loadings = pd.DataFrame(
    pca.components_.T,
    index=X.columns,
    columns=["PC1_loading", "PC2_loading"]
)

print("\nExplained variance ratio:")
print(f"PC1: {explained[0]:.4f}")
print(f"PC2: {explained[1]:.4f}")
print(f"Cumulative: {explained.sum():.4f}")

print("\nLoadings:")
print(loadings.sort_values("PC1_loading", key=lambda s: s.abs(), ascending=False))

# =========================
# 6. SAVE OUTPUTS
# =========================

output_full = pca_df.join(X)
output_full.to_csv("fred_pca_2000_2010_output.csv")
loadings.to_csv("fred_pca_loadings.csv")

# =========================
# 7. PLOT 1: TRAJECTORY IN LATENT SPACE
# =========================

plt.figure(figsize=(10, 8))

# Plot line
plt.plot(pca_df["PC1"], pca_df["PC2"], linewidth=1.8)

# Mark every January with a label
yearly_points = pca_df[pca_df.index.month == 1]
for dt, row in yearly_points.iterrows():
    plt.scatter(row["PC1"], row["PC2"], s=35)
    plt.text(row["PC1"] + 0.03, row["PC2"] + 0.03, str(dt.year), fontsize=9)

# Highlight Sept 2008 if present
target_date = pd.Timestamp("2008-09-30")
closest_idx = pca_df.index.get_indexer([target_date], method="nearest")[0]
target_row = pca_df.iloc[closest_idx]
target_dt = pca_df.index[closest_idx]

plt.scatter(target_row["PC1"], target_row["PC2"], s=90, marker="x")
plt.text(
    target_row["PC1"] + 0.05,
    target_row["PC2"] + 0.05,
    f"{target_dt.strftime('%Y-%m')} Lehman phase",
    fontsize=9
)

plt.axhline(0, linestyle="--", linewidth=0.8)
plt.axvline(0, linestyle="--", linewidth=0.8)

plt.title("Financial System Trajectory in PCA Latent Space (2000–2010)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("trajectory_pca_financial_crisis.png", dpi=300)
plt.show()

# =========================
# 8. PLOT 2: LOADINGS
# =========================

fig, axes = plt.subplots(2, 1, figsize=(11, 9))

loadings["PC1_loading"].sort_values().plot(kind="barh", ax=axes[0])
axes[0].set_title("PC1 Loadings")
axes[0].set_xlabel("Loading")

loadings["PC2_loading"].sort_values().plot(kind="barh", ax=axes[1])
axes[1].set_title("PC2 Loadings")
axes[1].set_xlabel("Loading")

plt.tight_layout()
plt.savefig("pca_loadings_financial_crisis.png", dpi=300)
plt.show()

# =========================
# 9. PLOT 3: TIME EVOLUTION OF PCS
# =========================

plt.figure(figsize=(12, 6))
plt.plot(pca_df.index, pca_df["PC1"], label="PC1", linewidth=1.8)
plt.plot(pca_df.index, pca_df["PC2"], label="PC2", linewidth=1.8)
plt.axvline(pd.Timestamp("2008-09-15"), linestyle="--", linewidth=1.0)
plt.title("Time Evolution of Principal Components")
plt.xlabel("Date")
plt.ylabel("Component Score")
plt.legend()
plt.tight_layout()
plt.savefig("pcs_time_series_financial_crisis.png", dpi=300)
plt.show()