import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Find the uploaded CSVs and load each one as one indicator
files = sorted(glob.glob("/content/*.csv"))
if not files:
    raise FileNotFoundError("Upload your CSVs with the Files panel first.")

def load_indicator(path, short_name):
    d = pd.read_csv(path)
    # The value column is the last one that isn't Entity, Code or Year
    value_col = [c for c in d.columns if c not in ("Entity", "Code", "Year")][-1]
    # Keep real countries only: their codes are 3 letters, and regional totals are not
    d = d[d["Code"].fillna("").str.len() == 3]
    return d[["Entity", "Year", value_col]].rename(columns={value_col: short_name})

# The outcome is the file with "mort" in its name; everything else is an indicator
outcome_file = [f for f in files if "mort" in f.lower()][0]
OUT = "outcome"
frames = [load_indicator(outcome_file, OUT)]
indicators = []
for i, f in enumerate(f for f in files if f != outcome_file):
    name = f"ind{i}"
    frames.append(load_indicator(f, name))
    indicators.append(name)
print("Outcome file:", outcome_file)
print("Indicator files:", [f for f in files if f != outcome_file])

# 2. Merge everything on country and year
df = frames[0]
for d in frames[1:]:
    df = df.merge(d, on=["Entity", "Year"], how="outer")
df = df.sort_values(["Entity", "Year"]).reset_index(drop=True)
print("Rows, countries, years:", len(df), df["Entity"].nunique(), df["Year"].min(), df["Year"].max())

# 3. Fill short gaps (at most 2 years) within each country
cols = [OUT] + indicators
df[cols] = df.groupby("Entity")[cols].transform(lambda s: s.ffill(limit=2))

# 4. Last year's values become the inputs, so the model forecasts one year ahead
g = df.groupby("Entity")
lag_cols = []
for c in cols:
    df[f"{c}_lag1"] = g[c].shift(1)
    lag_cols.append(f"{c}_lag1")
feat_cols = lag_cols

data = df.dropna(subset=[OUT] + feat_cols).copy()

# 5. Split by time, never at random: the last 5 years are the test set
test_start = data["Year"].max() - 4
train = data[data["Year"] < test_start]
test = data[data["Year"] >= test_start].copy()
print(f"Train: {len(train)} rows before {test_start}, test: {len(test)} rows from {test_start}")

# 6. Train the model and make predictions for the test years
model = LinearRegression().fit(train[feat_cols], train[OUT])
test["model_pred"] = model.predict(test[feat_cols])
test["baseline_pred"] = test[f"{OUT}_lag1"]  # baseline: this year equals last year

# 7. Compare errors (mean absolute error: average gap from the true value)
mae_model = (test[OUT] - test["model_pred"]).abs().mean()
mae_base = (test[OUT] - test["baseline_pred"]).abs().mean()
print(f"MAE model:    {mae_model:.2f}")
print(f"MAE baseline: {mae_base:.2f}")

# 8. Top 10 highest predicted risk in the last test year
latest = test[test["Year"] == test["Year"].max()]
top10 = latest.nlargest(10, "model_pred")[["Entity", "Year", "model_pred", OUT]]
print(top10.round(1).to_string(index=False))
