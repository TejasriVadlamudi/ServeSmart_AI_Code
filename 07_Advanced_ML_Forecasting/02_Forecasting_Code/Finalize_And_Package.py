from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,json,csv,shutil
import pandas as pd,numpy as np,openpyxl
R=Path(__file__).resolve().parents[1];E=R/'06_Model_Evaluation';D=R/'09_Documentation';F=R/'08_Results_and_Findings';rep=R/'09_Reproduction_Runs/Reproducibility_Check'
# Reproduction is numerical, not merely identical filenames.
checks=[]
for name in ['Overall_Model_Comparison.csv','All_Group_Performance.csv']:
 a=pd.read_csv(E/name);b=pd.read_csv(rep/'06_Model_Evaluation'/name);pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-9,atol=1e-9);checks.append(name)
a=pd.read_csv(E/'Prediction_Ledger/All_Test_Predictions.csv');b=pd.read_csv(rep/'06_Model_Evaluation/Prediction_Ledger/All_Test_Predictions.csv');pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-9,atol=1e-9)
(E/'Reproducibility_Check.json').write_text(json.dumps({'Status':'PASS','Environment':'New isolated Python3.12 environment installed from pinned requirements inside this new project; two complete executions in separate result folders.','AllTestPredictionsMatch':True,'PredictionRows':len(a),'MetricTablesMatched':checks,'Tolerance':'1e-9'},indent=2))
# All 204 validation trials include newly executed B configurations; audit row counts.
v=pd.read_csv(E/'All_Validation_Experiments.csv');assert len(v)==204
for level in ['Hourly','Daily']:
 assert set(v[v.ForecastLevel.eq(level)].Model)=={'Seasonal Naive','Historical Average','Ridge','Random Forest','XGBoost','LightGBM','Prophet'}
 for model in ['Random Forest','XGBoost','LightGBM']:assert set(v[(v.ForecastLevel==level)&v.Model.eq(model)].Config)=={0,1,2,3,4}
# Verify exported workbook cell values and cached formula results using read-only access.
wb=openpyxl.load_workbook(E/'Overall_Model_Comparison.xlsx',data_only=True,read_only=True);official=pd.read_csv(E/'Overall_Model_Comparison.csv')
assert len(wb.sheetnames)==10
for level in ['Hourly','Daily']:
 ws=wb[level];subset=official[official.ForecastLevel.eq(level)].reset_index(drop=True)
 for i,r in enumerate(subset.itertuples(),6):
  assert ws.cell(i,1).value==r.Model
  for col,key in [(3,'Observations'),(4,'ValidationRank'),(5,'ValidationMAE'),(6,'MAE'),(7,'RMSE'),(8,'WAPE')]:assert np.isclose(ws.cell(i,col).value,getattr(r,key))
ws=wb['Overview']
for row in [6,7]:assert np.isclose(ws.cell(row,8).value,(ws.cell(row,7).value-ws.cell(row,4).value)/ws.cell(row,7).value)
wb.close();(E/'Workbook_Value_Verification.json').write_text(json.dumps({'Status':'PASS','Sheets':10,'All14ComparisonRowsMatchCSV':True,'BothCachedImprovementFormulasMatch':True,'VisualReview':'Rendered all ten sheets; labels wrapped and numbers/units checked. Excel native interactions were not controlled.'},indent=2))
# Original source packages and designated dataset remain byte-identical.
for m in json.loads((D/'Original_Package_Hashes.json').read_text()):assert hashlib.sha256(Path(m['Original']).read_bytes()).hexdigest()==m['SHA256']
source=Path.home()/'Desktop/ServeSmart_FinalFlatfile_Hourly_ML.csv';assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((R/'01_Approved_Data/Approved_ML_Flat_File.csv').read_bytes()).digest()
(D/'Originals_Unchanged_Check.json').write_text(json.dumps({'Status':'PASS','BothOriginalPackageHashesUnchanged':True,'OriginalDesignatedFlatFileUnchanged':True,'ExistingBaselineTableauAndPresentationFolders':'No write operations were performed on them.'},indent=2))
# Short, polished front page makes the strict status easy to understand.
(R/'OPEN_THIS_FIRST.txt').write_text('''SERVESMART AI — CONSOLIDATED FORECASTING VERSION

The new experiments and verification are complete. Both original packages and
all existing baseline, Tableau and presentation folders remain unchanged.

Read 08_Results_and_Findings/Final_Findings.md and Remaining_Blockers.txt.
Open 06_Model_Evaluation/Overall_Model_Comparison.xlsx for the ten-tab report.
Use 10_Final_Submission/ServeSmart_Advanced_Forecasting_Ready_To_Submit.zip
as the complete review package.

Validation-selected models:
Hourly: Historical Average. Daily: Random Forest.
Seasonal Naive has lower May errors; the report retains the validation choices.
All seven methods ran at both levels, including LightGBM and Prophet.

STRICT VERDICT: NOT YET READY to claim that every requirement is fully met.
The runnable research evidence is complete. Missing timed enrollment sources,
historical availability evidence, an outlet operating calendar and Package B's
omitted final runner prevent an unconditional completion claim.
Do not confuse the requested folder name with certification of submission readiness.
''')
(F/'Remaining_Blockers.txt').write_text('''STRICT VERDICT: NOT YET READY FOR AN UNCONDITIONAL COMPLETION CLAIM

Completed: all seven hourly/daily methods; 204 validation trials; documented
Package B settings rerun on safe features; 14 overall and 623 grouped scores;
selected-model SHAP/explanations; model reload and prepared-batch inference;
two full runs with matching results; the comparison workbook and final ZIP.

Evidence still needed to close the partial requirements:
1. The exact weekly class-profile generator and effective-date/section rules,
   plus historical availability timestamps. The current master has blank class
   times. Related timed Tableau evidence is included for audit but does not
   fully reproduce the supplied weekly profile under the checked methods.
   These registrations are not measured attendance.
2. An outlet opening/closure calendar. Current daily targets are recorded-day
   units and cannot be certified as complete future operating-day demand.
3. Package B's missing run_ml_models.py and its exact preprocessing/daily
   aggregation logic, if an exact original end-to-end reproduction is required.
4. A genuinely new holdout for prospective performance. May was already used
   in previous project development; no rerun can erase that earlier exposure.

The package is suitable for a transparent research review with these limits.
It must not be described as a fully verified live operational forecaster.
''')
# Explain the actual selected model with natural wording, without overstating performance.
(F/'Presentation_And_Mock_Interview_Explanation.txt').write_text('''I developed one reproducible forecasting workflow from our corrected ETL and
ML flat file. I verified the targets and rebuilt historical features using only
information from earlier dates. I excluded Spring Break labels, unsafe database
identifiers and future actual weather. I compared Seasonal Naive, Historical
Average, Ridge, Random Forest, XGBoost, LightGBM and Prophet at hourly and daily
levels. I also reran Tejasri's documented tree settings using the safe features.

Models were fitted on January–February and selected on March–April validation.
Historical Average won hourly validation; Random Forest won daily validation.
All methods were evaluated on identical May observations. Seasonal Naive had
lower May errors, so I reported the generalization limitation instead of
changing the winner after inspecting the test scores. The saved predictions,
independent metric checks and a second matching run support reproducibility.

I also documented what the sources cannot establish: the original timed class
schedule and historical enrollment availability are missing, opening-hour
coverage is unverified, and Package B omits its final runner. These limitations
must be resolved before claiming that every requirement or live integration is
fully verified. The model outputs are POS units, not people or attendance.
''')
# Model files and public per-level outputs are all present.
registry=json.loads((R/'03_Trained_Models/Model_Registry.json').read_text());assert len(registry)==14
for m in registry:assert (R/m['ModelFile']).is_file()
for folder in ['04_Hourly_Predictions','05_Daily_Predictions']:
 assert list((R/folder).glob('*Predictions.csv'))
# Put artifact-tool's verbose inspect log with verification, away from headline outputs.
for p in E.glob('*.inspect.ndjson'):shutil.move(str(p),D/p.name)
excluded={'.venv','.cache','__pycache__','node_modules','09_Reproduction_Runs','10_Final_Submission','.DS_Store'}
files=[p for p in sorted(R.rglob('*')) if p.is_file() and not any(q in excluded for q in p.relative_to(R).parts) and p.name!='File_Inventory.csv']
with (D/'File_Inventory.csv').open('w',newline='') as h:
 writer=csv.DictWriter(h,fieldnames=['RelativePath','Bytes','SHA256']);writer.writeheader()
 for p in files:writer.writerow(dict(RelativePath=str(p.relative_to(R)),Bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
files.append(D/'File_Inventory.csv');dest=R/'10_Final_Submission';dest.mkdir(exist_ok=True);archive=dest/'ServeSmart_Advanced_Forecasting_Ready_To_Submit.zip'
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(Path(R.name)/p.relative_to(R)))
with ZipFile(archive) as z:
 assert z.testzip() is None
 for p in files:assert hashlib.sha256(z.read(str(Path(R.name)/p.relative_to(R)))).digest()==hashlib.sha256(p.read_bytes()).digest()
(dest/'ZIP_Verification.json').write_text(json.dumps({'Status':'PASS','Files':len(files),'Bytes':archive.stat().st_size,'SHA256':hashlib.sha256(archive.read_bytes()).hexdigest(),'AllArchiveFilesByteMatched':True,'Excluded':'Virtual environment, caches, authoring dependencies, duplicate reproduction staging and archive itself','Readiness':'See explicit strict verdict and remaining source-evidence blockers; ZIP integrity is not full requirement satisfaction.'},indent=2))
print('PASS: 204 trials; two runs match; ten-sheet XLSX values verified; originals unchanged; final ZIP',len(files),'files,',archive.stat().st_size,'bytes.')
