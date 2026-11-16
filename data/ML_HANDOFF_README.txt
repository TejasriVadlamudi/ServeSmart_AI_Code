ServeSmart Updated ML Flat File

Grain: one observed dining location, date, and hour.
Rows: 2,032.
Target: Target_HourlyUnits.
Source: approved final dimensional ETL outputs.

Use only the 22 predictors in Model_Input_List.csv to reproduce
the completed modeling experiment.

Identifiers, sales amounts, context fields, split labels, and
target-review flags are not additional model predictors.

Historical missing values remain blank. Apply the saved model
pipelines to handle preprocessing; do not replace blanks with
zero automatically.

Retain the original negative target in the file. As in the
completed experiment, exclude negative targets from ML fitting.

Train: January–February 2026.
Validation: March–April 2026.
Test: May 2026.
Final ML fitting: January–April, excluding negative targets.

Weather, event, and class-context fields are supporting columns.
They were not included in the completed models' 22 predictors.

Model_Data_Comparison.csv confirms agreement with the completed
modeling dataset for every target, input, and split label.
