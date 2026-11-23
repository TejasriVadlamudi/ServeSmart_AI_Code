from pathlib import Path
R=Path(__file__).resolve().parents[1];p=R/'02_Forecasting_Code/Forecasting_Pipeline.py';s=p.read_text()
s=s.replace("'Prev28DayCount']","'Prev28DayCount','HistoryClockMean','HistoryClockCount','Recent28WeekdayMean','Recent56WeekdayMean','LastObservedComparable','ComparableAgeDays','Recent7ClockMean','Recent28ClockMean','Lag7MinusLag14','Lag7To14Ratio']",1)
needle="  features.append(v)"
s=s.replace(needle,"""  clock=prior[prior.Hour.eq(hour)] if hourly else prior
  v.update(HistoryClockMean=clock.Actual.mean(),HistoryClockCount=clock.Actual.count(),Recent28WeekdayMean=same.loc[same.Date.ge(date-pd.Timedelta(days=28)),'Actual'].mean(),Recent56WeekdayMean=same.loc[same.Date.ge(date-pd.Timedelta(days=56)),'Actual'].mean(),LastObservedComparable=clock.Actual.iloc[-1] if len(clock) else np.nan,ComparableAgeDays=(date-clock.Date.iloc[-1]).days if len(clock) else np.nan,Recent7ClockMean=clock.loc[clock.Date.ge(date-pd.Timedelta(days=7)),'Actual'].mean(),Recent28ClockMean=clock.loc[clock.Date.ge(date-pd.Timedelta(days=28)),'Actual'].mean(),Lag7MinusLag14=v['Lag7Day']-v['Lag14Day'],Lag7To14Ratio=v['Lag7Day']/(abs(v['Lag14Day'])+1))
  features.append(v)""")
s=s.replace("def estimator(name,config):\n", """def estimator(name,config):
 # Package B's two documented settings per tree family, re-evaluated on SAFE features.
 if config>=3 and name in ['Random Forest','XGBoost','LightGBM']:
  conf=config-3
  if name=='Random Forest':return RandomForestRegressor(n_estimators=[200,300][conf],max_depth=[8,None][conf],max_features=.8,min_samples_leaf=2,n_jobs=2,random_state=42)
  if name=='XGBoost':return XGBRegressor(n_estimators=[200,300][conf],max_depth=[4,3][conf],learning_rate=.05,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',n_jobs=2,random_state=42)
  if name=='LightGBM':return LGBMRegressor(n_estimators=[200,300][conf],num_leaves=[15,31][conf],learning_rate=[.05,.03][conf],subsample=.9,colsample_bytree=.9,n_jobs=2,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
""")
s=s.replace("configs=range(2) if name=='Prophet' else range(3)","configs=range(2) if name=='Prophet' else range(3) if name=='Ridge' else range(5)")
s=s.replace("def main():",'''def validation_daily_approaches(root,frames):
 # Compare every dedicated daily model and observed-hour aggregation on identical validation days.
 v=pd.read_csv(root/'06_Model_Evaluation/Prediction_Ledger/Validation_Predictions.csv',parse_dates=['Date'])
 h=v[v.ForecastLevel.eq('Hourly')];d=v[v.ForecastLevel.eq('Daily')]
 expected=frames['Hourly'].query('DataSplit=="Validation"').groupby(['Location','Date']).size().rename('ExpectedRecordedHours')
 ag=h.groupby(['Model','Location','Date'],as_index=False).agg(Prediction=('Prediction',lambda a:a.sum(min_count=1)),PredictedRecordedHours=('Prediction','count')).merge(expected,on=['Location','Date'])
 ag=ag[ag.PredictedRecordedHours.eq(ag.ExpectedRecordedHours)]
 actual=frames['Daily'].query('DataSplit=="Validation"')[['Location','Date','Actual']]
 ag=ag.merge(actual,on=['Location','Date']);ag['Approach']='Sum_Recorded_Hourly'
 direct=d[['Model','Location','Date','Actual','Prediction']].copy();direct['Approach']='Dedicated_Daily'
 both=pd.concat([direct,ag],ignore_index=True);count=both[both.Prediction.notna()].groupby(['Location','Date']).size();keys=count[count.eq(14)].reset_index()[['Location','Date']];shared=both.merge(keys,on=['Location','Date'])
 rows=[]
 for (approach,model),g in shared.groupby(['Approach','Model']):rows.append(dict(Approach=approach,Model=model,**score(g.Actual,g.Prediction)))
 pd.DataFrame(rows).to_csv(root/'06_Model_Evaluation/Validation_Daily_Approach_Comparison.csv',index=False)
 shared.to_csv(root/'06_Model_Evaluation/Validation_Daily_Approach_Predictions.csv',index=False)
 # Reliability gate is predeclared: missing true opening-hour coverage prevents a full-day sum claim.
 jsonout(root/'03_Trained_Models/Daily_Approach_Lock_Before_Test.json',dict(OfficialApproach='Dedicated_Daily',SelectionPolicy='Compare common validation MAE, but require known future hour-set coverage before using hourly sums as an operational daily forecast. Opening-hour calendar unavailable; retain dedicated recorded-day model. No test score changes this reliability gate.',CommonValidationDays=len(keys),FullOperatingHourCoverageVerified=False))

def publish_level_outputs(root):
 ledger=root/'06_Model_Evaluation/Prediction_Ledger'
 import shutil
 for level,folder in [('Hourly','04_Hourly_Predictions'),('Daily','05_Daily_Predictions')]:
  dest=root/folder;dest.mkdir(parents=True,exist_ok=True)
  for p in ledger.glob(level+'*.csv'):shutil.copy2(p,dest/p.name)
  for name in ['Validation_Predictions.csv','All_Test_Predictions.csv']:
   d=pd.read_csv(ledger/name);d[d.ForecastLevel.eq(level)].to_csv(dest/name,index=False)

def main():''')
s=s.replace("chosen=train_validate(root,frames);pred,comparison,detail,registry,models=evaluate", "chosen=train_validate(root,frames);validation_daily_approaches(root,frames);pred,comparison,detail,registry,models=evaluate")
s=s.replace("emit(comparison[['ForecastLevel'", "publish_level_outputs(root)\n emit(comparison[['ForecastLevel'")
p.write_text(s)
# Record explicit, pre-May search design and the distinction between borrowed settings and unverified original implementation.
(R/'03_Trained_Models/Predeclared_Search_Policy.json').write_text(__import__('json').dumps({'Split':'Jan-Feb Train;Mar-Apr Validation;May historical benchmark','Selection':'Lowest common-validation MAE; RMSE tie-break. No May-driven parameter changes.','NewHistory':'Recent28/56 same-weekday means; pooled same-clock history; last prior comparable observation and its age; recent7/28-clock means; lag7-versus14 trend. All Date<forecastDate.','BorrowedPackageBSettings':'Config3 and4 are the two documented settings per RF/XGBoost/LightGBM. n_jobs constrained to2 and LightGBM deterministic execution. Original B runner missing and its unsafe features are not reproduced as approved forecasts.','Schedule':'Weekly ungated profile; conditional as-of snapshot, not actual attendance.','DailyReliabilityGate':'Without outlet operating calendar, retain dedicated recorded-day model; separately report validation aggregation comparison on common days.','TestCaveat':'May has prior project exposure; never call it prospectively untouched.'},indent=2))
print('Adapted relative folder paths, safe history, Package B parameter sets and daily validation comparison.')
