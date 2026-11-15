# Write-up: country_risk.py

This script builds a simple forecasting model for a country-level outcome, most likely a mortality rate given the filename convention. It combines several yearly indicator datasets, trains a linear regression to predict next year's outcome for each country, checks whether the model beats a naive "same as last year" guess, and lists the ten countries with the highest predicted values in the most recent test year.

## Purpose

The script answers one question: given what a country looked like last year on several indicators, such as income, health spending, or education, how well can we predict its outcome this year? The name `country_risk` suggests that high predicted values are read as high risk, so the top-ten list identifies the countries the model expects to have the worst outcomes.

## Setup and file discovery

The script expects CSV files in `/content/`, the default upload folder in Google Colab. It collects every CSV there and stops with a clear error if none are found. One file is designated as the outcome by finding the first filename containing "mort" (case-insensitive). Every other CSV is treated as an explanatory indicator. This convention is convenient but fragile: a filename without "mort" causes an `IndexError`, and the script will not warn you if two files contain that string.

## Loading and cleaning

Each CSV is read by the `load_indicator` function. The script assumes the value of interest is the last column that is not `Entity`, `Code`, or `Year`, which matches the layout of Our World in Data exports. It then keeps only rows whose `Code` is exactly three letters. This removes regional and world aggregates, which either have no code or have longer codes such as `OWID_WRL`, so the model works only with individual countries. Each file is reduced to three columns (country, year, value), and the value column is renamed to a short internal name.

The outcome keeps the name `outcome`. The indicators are named `ind0`, `ind1`, and so on, in the order the files were found. The script prints which file is which, so you can match the names back to the data.

## Merging and filling gaps

All the cleaned tables are joined on country and year with an outer merge, so a country-year appears even if only some files cover it. The data is then sorted by country and year. Within each country, short gaps are filled by carrying the last known value forward, but only for up to two consecutive years. Longer gaps stay empty, which prevents stale values from being carried too far.

## Building the features

For the outcome and each indicator, the script creates a one-year lagged copy by shifting values down within each country. Each row therefore holds last year's outcome and indicators, and the goal is to predict this year's outcome from them. Rows missing the outcome or any lagged feature are dropped.

One subtlety is worth knowing. `shift(1)` moves by row position, not by calendar year. If a country is missing an entire year in the merged data, its "lag" refers to the most recent available year, which may be two or more years earlier. A more careful version would reindex each country to a complete annual series before shifting.

## Train and test split

The split is by time rather than random. The last five years in the data form the test set, and everything earlier is training data. This matters because a random split would let the model train on future years and make the test results look better than they would in real use.

## Modeling and baseline

A scikit-learn `LinearRegression` is fitted on the training rows to predict the outcome from its lagged features. The script then compares it with a simple baseline that predicts this year's value equals last year's. This baseline is a strong benchmark for slow-moving indicators, and a model that cannot beat it adds little value.

Performance is measured with mean absolute error (MAE), the average size of the gap between predicted and actual values, in the same units as the outcome. The script prints both MAEs. If the model's MAE is lower than the baseline's, it adds predictive value over persistence. If not, the indicators are not helping.

## Output: top ten countries

The script takes the most recent test year, sorts countries by predicted outcome, and prints the ten highest with their predicted and actual values. Because the predictions are forward-looking, the most recent test year gives the most meaningful forecasts, though it is the only year printed.

## Limitations

The model is linear and uses only one year of history, so it cannot capture trends or non-linear effects. It has no country-specific terms, so it cannot learn that some countries sit persistently above or below what their indicators predict. It also ignores the uncertainty around each prediction, so the ranking should be read as a point estimate rather than a firm verdict. Finally, the model captures correlation useful for prediction; it does not establish that any indicator causes changes in the outcome.

## Summary

The script is a compact, well-structured pipeline: load and clean several country-year datasets, merge and fill them, create lagged features, train on earlier years and test on recent ones, compare against a naive baseline, and rank countries by predicted outcome. Its main weaknesses are the filename-based file detection, the row-based lag that can misalign years, and the lack of uncertainty estimates. Addressing those would make it more robust for real analysis.
