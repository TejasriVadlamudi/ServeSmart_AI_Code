"""Audit A/B records and claims. B saved metrics remain external historical results."""
from pathlib import Path
import json,csv,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];B=R/'09_Documentation/Package_B_Reference';final=B/'FINAL_Advanced_ML_Outputs_20261122/outputs';src=R/'01_Approved_Data/Approved_ML_Flat_File.csv';h=pd.read_csv(src,parse_dates=['Date'])
inputB=next(B.rglob('ServeSmart_FinalFlatfile_Hourly_ML(1).csv'));b=pd.read_csv(inputB,parse_dates=['Date']);keys=['Location','Date','Hour'];j=h[keys+['Target_HourlyUnits']].merge(b[keys+['Target_HourlyUnits']],on=keys,how='outer',suffixes=('_A','_B'),indicator=True,validate='one_to_one');assert j._merge.eq('both').all() and np.allclose(j.Target_HourlyUnits_A,j.Target_HourlyUnits_B)
features=pd.read_csv(final/'feature_and_split_manifest.csv');audit=[]
for r in features.itertuples():
 for feature in r.Features.split('; '):
  risk='No identified timing issue from name alone; construction still needs verification'
  if feature in ['ClassKey','WeatherKey','EventKey']:risk='Exclude identifier: not a substantive predictor; may act as time/source proxy'
  if feature in ['RegularClassDay','ClassesActive','EstimatedStudentsOnCampus','ClassesStarting','ClassesEnding','HasClass','ScheduledStudents_DailyPeak']:risk='Date-gated schedule/break/missing-zero proxy; use independently documented raw schedule profile instead'
  if feature in ['TemperatureF','HumidityPct','PrecipitationIn','RainFlag','CloudCoverPct','WindSpeedKmh']:risk='Realized forecast-day weather; no archived forecast-time weather feed in package'
  if feature in ['HasEvent','CampusEventCount','EventOccurrenceCount','EventFoodCount','AllDayEventCount','EventTypeCount']:risk='Same-day event context requires publication/as-of verification; absent in supplied evidence'
  if r.ForecastLevel=='Daily' and feature in ['Hour','ClassesActive','TemperatureF']:risk+='; daily aggregation rule not recoverable because final runner missing; observed-hour averaging may reveal target coverage'
  audit.append(dict(ForecastLevel=r.ForecastLevel,SourceFeature=feature,Audit=risk))
pd.DataFrame(audit).to_csv(R/'06_Model_Evaluation/Package_B_Feature_Audit.csv',index=False)
p=pd.read_csv(final/'predictions_test.csv',parse_dates=['Date']);rows=[]
# Common rows match the official baseline eligibility definition reconstructed from targets.
lookup={(r.Location,r.Date,int(r.Hour)):r.Target_HourlyUnits for r in h.itertuples()}
daily=h.groupby(['Location','Date']).Target_HourlyUnits.sum()
for (level,model),g in p.groupby(['ForecastLevel','Model']):
 if level=='Hourly':
  expected=[lookup[(r.Location,r.Date,int(r.Hour))] for r in g.itertuples()];common=np.array([(r.Location,r.Date-pd.Timedelta(days=7),int(r.Hour)) in lookup for r in g.itertuples()])
 else:expected=[daily.loc[(r.Location,r.Date)] for r in g.itertuples()];common=np.array([(r.Location,r.Date-pd.Timedelta(days=7)) in daily.index for r in g.itertuples()])
 assert np.allclose(g.Actual,expected)
 for scope,mask in [('All_B_saved_test_rows',np.ones(len(g),bool)),('Common_previous_week_available_keys',common)]:
  z=g.iloc[np.flatnonzero(mask)];err=z.Predicted-z.Actual;rows.append(dict(Source='Package B stored predictions; not a safe rerun',ForecastLevel=level,Model=model,Scope=scope,Observations=len(z),MAE=abs(err).mean(),RMSE=np.sqrt(np.mean(err**2)),WAPE=100*abs(err).sum()/abs(z.Actual).sum()))
pd.DataFrame(rows).to_csv(R/'06_Model_Evaluation/Package_B_Stored_Predictions_Recalculated.csv',index=False)
pd.read_csv(final/'validation_tuning.csv').to_csv(R/'06_Model_Evaluation/Package_B_Documented_Configurations.csv',index=False)
# Determine what the supplied HistoricalSameHour_Avg actually computes.
x=h.sort_values(['Location','Date','Hour']).copy();pool=x.groupby(['Location','Hour']).Target_HourlyUnits.transform(lambda t:t.shift().expanding().mean());strict=x.groupby(['Location','Weekday','Hour']).Target_HourlyUnits.transform(lambda t:t.shift().expanding().mean())
summary={'SameApprovedHourlyTargets':True,'Rows':len(j),'B_FinalRunnerPresent':any(p.name=='run_ml_models.py' for p in B.rglob('*.py')),'FlatFileHistoricalSameHourAvg_matches_pooled_location_clock':bool(np.allclose(x.HistoricalSameHour_Avg,pool,equal_nan=True,atol=1e-6)),'FlatFileHistoricalSameHourAvg_matches_same_weekday_clock':bool(np.allclose(x.HistoricalSameHour_Avg,strict,equal_nan=True,atol=1e-6)),'B_Python':'3.9.6','B_DocumentedVersions':pd.read_csv(final/'ml_dependency_status.csv').fillna('').to_dict('records'),'Note':'Exact original end-to-end rerun is not recoverable from B because run_ml_models.py is omitted. Documented model parameters are rerun on explicit safe predictors in the new official pipeline; stored B scores are reference only. WAPE in B is fractional; comparisons here express percent. No measured causal allocation of the A-B score gap is claimed.'}
(R/'06_Model_Evaluation/Source_Package_Reconciliation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
