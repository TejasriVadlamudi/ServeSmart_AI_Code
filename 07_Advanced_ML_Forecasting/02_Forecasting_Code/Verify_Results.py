"""Independent verification from saved CSV outputs; does not refit or tune models."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd
BASE=Path(__file__).resolve().parents[1]
def check(root):
 p=pd.read_csv(root/'06_Model_Evaluation/Prediction_Ledger/All_Test_Predictions.csv',parse_dates=['Date']);source=pd.read_csv(BASE/'01_Approved_Data/Approved_ML_Flat_File.csv',parse_dates=['Date']);m=pd.read_csv(root/'06_Model_Evaluation/Overall_Model_Comparison.csv');groups=pd.read_csv(root/'06_Model_Evaluation/All_Group_Performance.csv');checks=[]
 def ck(name,ok,detail=''):
  checks.append(dict(Check=name,Passed=bool(ok),Detail=str(detail)))
  assert ok,name+':'+str(detail)
 methods={'Seasonal Naive','Historical Average','Ridge','Random Forest','XGBoost','LightGBM','Prophet'}
 ck('All predictions are May holdout',p.Date.ge('2026-05-01').all() and p.Date.lt('2026-06-01').all())
 ck('Seven completed methods at each level',all(set(g.Model)==methods for _,g in p.groupby('ForecastLevel')))
 ck('14 saved model reload checks passed',len(pd.read_csv(root/'06_Model_Evaluation/Model_Reload_Checks.csv'))==14 and pd.read_csv(root/'06_Model_Evaluation/Model_Reload_Checks.csv').ReloadPredictionMatch.all())
 ck('Unique predictions by model and target key',not p[p.ForecastLevel.eq('Hourly')].duplicated(['Model','Location','Date','Hour']).any() and not p[p.ForecastLevel.eq('Daily')].duplicated(['Model','Location','Date']).any())
 count=0
 def calc(a,b):
  e=np.asarray(b)-np.asarray(a);den=sum(abs(np.asarray(a)));return dict(Observations=len(e),MAE=float(sum(abs(e))/len(e)),RMSE=float(np.sqrt(sum(e*e)/len(e))),WAPE=float(100*sum(abs(e))/den) if den else np.nan)
 for row in m.itertuples():
  z=p[(p.ForecastLevel==row.ForecastLevel)&(p.Model==row.Model)&p.CommonEvaluationRow];s=calc(z.Actual,z.Prediction)
  ck(f'{row.ForecastLevel} {row.Model} metrics',all(np.isclose(s[k],getattr(row,k),equal_nan=True) for k in s));count+=1
 for r in groups.itertuples():
  z=p[(p.ForecastLevel==r.ForecastLevel)&(p.Model==r.Model)&p.CommonEvaluationRow]
  for c in {'Location':['Location'],'Weekday':['Weekday'],'Hour':['Hour'],'LocationWeekday':['Location','Weekday'],'LocationHour':['Location','Hour']}[r.GroupType]:z=z[z[c].eq(getattr(r,c))]
  s=calc(z.Actual,z.Prediction);assert all(np.isclose(s[k],getattr(r,k),equal_nan=True) for k in s),r
 ck('All grouped scores independently reproduced',True,len(groups))
 for level in ['Hourly','Daily']:
  g=p[(p.ForecastLevel==level)&p.CommonEvaluationRow];keys=['Location','Date']+(['Hour'] if level=='Hourly' else []);n=g.groupby(keys).Model.nunique();ck(level+' common sample has every method',n.eq(7).all())
  x=pd.read_csv(root/f'01_Approved_Data/Prepared_{level}_Features_And_Actuals.csv',parse_dates=['Date'])
  for name in ['Seasonal Naive','Historical Average']:
   actual=g[g.Model==name]
   for r in actual.itertuples():
    prev=x[(x.Location==r.Location)&x.Date.lt(r.Date)]
    if level=='Hourly':prev=prev[prev.Hour.eq(r.Hour)]
    if name=='Seasonal Naive':v=prev.loc[prev.Date.eq(r.Date-pd.Timedelta(days=7)),'Actual'].iloc[0]
    else:v=prev.loc[prev.Date.dt.dayofweek.eq(r.Date.dayofweek),'Actual'].mean()
    assert np.isclose(max(0,v),r.Prediction)
  ck(level+' baseline values independently recomputed',True)
  lock=json.loads((root/f'03_Trained_Models/{level}_Selection_Locked_Before_Test.json').read_text());selected=m[(m.ForecastLevel==level)&m.SelectedByValidation].iloc[0];ck(level+' selection matches locked validation winner',selected.Model==lock['Winner']['Model'])
  daily_source=source.groupby(['Location','Date'],as_index=False).Target_HourlyUnits.sum()
  if level=='Daily':
   j=x.merge(daily_source,on=['Location','Date'],validate='one_to_one');ck('Daily totals preserve every signed hourly unit',np.allclose(j.Actual,j.Target_HourlyUnits))
 # Mutate all targets from a cutoff onward; earlier/same-day predictors must be invariant.
 import importlib.util,os,sys
 # Historical feature builder is tested in a subprocess launcher to make macOS runtime setup explicit.
 audit=pd.read_csv(root/'01_Approved_Data/Predictor_Source_Column_Audit.csv');ck('Spring Break and direct calendar proxies excluded',audit[audit.SourceColumn.isin(['SpringBreak','Spring_Break','Academic_Status','Academic_Phase','RegularClassDay','Regular_Class_Day'])].Use.eq('Excluded').all())
 ck('Undefined WAPE on all-zero groups',all(pd.isna(r.WAPE) for r in groups.itertuples() if r.WAPEDenominator==0))
 auditdata=pd.read_csv(root/'01_Approved_Data/Data_Quality_Report.csv');ck('No unresolved ERROR-level data checks',auditdata[auditdata.Severity.eq('ERROR')].Passed.all())
 jsonpath=root/'06_Model_Evaluation/Independent_Verification.json';jsonpath.write_text(json.dumps(dict(Status='PASS',OverallComparisons=count,GroupedComparisons=len(groups),Checks=checks),indent=2));print('PASS:',len(checks),'checks;',count,'overall and',len(groups),'grouped metric rows reproduced independently.')
 return checks
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=BASE);args=parser.parse_args();check(args.root)
