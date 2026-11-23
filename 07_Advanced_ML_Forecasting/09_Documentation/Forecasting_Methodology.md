# Methodology and reproducibility

Grain: hourlyLocation/Date/Hour and dailyLocation/Date. Target: recorded signed POS quantities;daily sums preserve the negative historical adjustment. One negative hourly training target is excluded from fitting;zero targets are retained. No missing hours are filled withzero and no current transaction-derived columns are predictors.

Cutoff:23:59:59 of the prior calendar date. Exact1/7/14-day lags,expanding strict same-weekday/hour baseline,pooled same-clock ML statistics,recent calendar windows,prior comparable demand/age and lag trend are computed withDate<forecastDate. Prior-day weather/events are the only external observation predictors;live data-ingestion delays are not independently verified. Weekly enrollment fields are rebuilt from the raw profile without break-date gating;unknown profile rows stayNaN until training-only median/missing-indicator preprocessing. Source snapshot as-of timing is unverified,so enrollment variants are explicitly conditional.

The deployed app handoff uses the selected models recovered from commit `64138acf`: hourly Random Forest and daily XGBoost. The verified historical May handoff contains 280 common hourly rows and 43 daily rows. Recomputed WAPE is 35.4479% hourly and 29.0414% daily. The app-facing files and model metadata preserve the original training configuration and forecast cutoff fields.

Candidate search: Package A framework settings plus two documented Package B settings per tree family,each over five context configurations. SeePredeclared_Search_Policy.json andAll_Validation_Experiments.csv;no May-driven changes. Bootstrap uses the OpenMP runtime already supplied by this environment's sklearn wheel onMac;no system installation modified. Models serialize sklearn pipelines,ProphetJSON orbaselineconfiguration. Original source archives are readonlycopies.

MAE=mean(abs(error));RMSE=sqrt(mean(error²));WAPE=100*sum(abs(error))/sum(abs(actual)),undefined withzero denominator. The exact recovery evidence is in `data/forecasting/Recovered_Model_Verification.json`. The older 37.37% hourly and 22.54% daily figures are not reproducible aggregate WAPE values and are not final claims. Seasonal Naive remains a lower-error May benchmark and is retained only as a comparison, not as the selected app model.

Daily aggregation is compared using identical validation days withcomplete RECORDED-hour sets. Unknown actual future opening hours prevent claiming full-day totals. A predeclared reliabilitygate retains the independent recorded-day model.
