# ServeSmart Advanced Forecasting — new consolidated version

Read08_Results_and_Findings/Final_Findings.md andRemaining_Blockers.txt before submitting. **Strict verdict: Not Yet Ready for an unconditional all-nine-requirements-complete claim.** All runnable seven-model experiments,reporting and verification are complete;missing enrollment/as-of/operating-calendar evidence and the omittedPackageB finalrunner are explicitly documented. The research package can be reviewed with those limitations. No original package was modified.

## Contents
01_Approved_Data:designated data,source snapshots,audits and prepared features.
02_Forecasting_Code:single pipeline,notebook,requirements,independent checks and inference utility.
03_Trained_Models:14 saved per-level model/configfiles,selection locks and registry.
04_Hourly_Predictions;05_Daily_Predictions:selected/all-model/validation historical outputs and website data.
06_Model_Evaluation:official comparisonCSV/XLSX,623grouped metrics,PackageA/B/newcomparison,dailyapproaches and checkrecords. Prediction_Ledger retains common tables shared across levels.
07_Feature_Importance_SHAP:current selected/best-advanced explanations.
08_Results_and_Findings:findings,blockers,presentation explanation.
09_Documentation:methods,ninegroupchecklist,sourcearchives,PackageBreference and logs.
10_Final_Submission:complete ZIP and checksums.

## Execute
UsePython3.12,not theMac systemPython3.9.
```
python3.12 -m venv .venv
.venv/bin/python -m pip install -r 02_Forecasting_Code/requirements.txt
.venv/bin/python 02_Forecasting_Code/Forecasting_Pipeline.py --output-run My_Run
.venv/bin/python 02_Forecasting_Code/Verify_Results.py --root 09_Reproduction_Runs/My_Run
```
The optional output-run folder is generated locally and excluded fromsubmission toavoid duplicate models. To verify supplied results without fitting,useVerify_Results.py. To regenerate feature stress checks and explanations,useVerify_Leakage_And_Explain.py. Compare_Source_Packages.py audits copiedA/B artifacts and explicitly marksB's stored scores as external history. Prepare_Final_Reports.py regenerates the report/workbookinputJSON. Build_Comparison_Workbook.mjs exports the workbook using@oai/artifact-tool;the Nodeauthoring dependency is separate from Pythontraining requirements. README in thatcodefolder explains this reporting-onlystep. Final packaging script regenerates inventory and theZIP.

The guided notebook launches the same official CLI;it does not duplicate training logic. The evidence logs record the completeisolated-environment rerun and independent verification. Current selection:hourlyHistoricalAverage,dailyRF. Both are chosen byvalidation,even thoughSeasonalNaive winsMay. All metrics are error measurements,not classificationaccuracy.
