"""ServeSmart: one-day-ahead, fixed-parameter/rolling-information backtest.
Run: python 02_Forecasting_Code/Forecasting_Pipeline.py --output-run RUN_NAME
Original source snapshots remain read-only. CSVs are generated reporting artifacts.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,logging,os,sys,time,warnings
from pathlib import Path
# On macOS, use the OpenMP runtime shipped in this environment's sklearn wheel.
# Relaunch once so dyld sees the project-local path before loading boosting libraries.
if sys.platform=='darwin' and __name__=='__main__' and not os.environ.get('SERVESMART_RUNTIME_READY'):
 import importlib.util
 spec=importlib.util.find_spec('sklearn')
 omp=Path(spec.origin).parent/'.dylibs' if spec else None
 if omp and (omp/'libomp.dylib').exists():
  env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(omp)+(':'+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '');env['SERVESMART_RUNTIME_READY']='1'
  os.execve(sys.executable,[sys.executable,*sys.argv],env)
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'.cache'))
import joblib,numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from prophet.serialize import model_to_json,model_from_json
import matplotlib.pyplot as plt
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
BASE=Path(__file__).resolve().parents[1]
SEED=42
METHODS=['Seasonal Naive','Historical Average','Ridge','Random Forest','XGBoost','LightGBM','Prophet']
SETS=['HistoricalOnly','HistoryCalendar','HistoryEnrollment','HistoryWeatherEvents','Full']
HISTORY=['Lag1Day','Lag7Day','Lag14Day','HistoryWeekdayMean','HistoryWeekdayCount','PrevDayTotal','Prev7DayMean','Prev7DayCount','Prev28DayMean','Prev28DayCount','HistoryClockMean','HistoryClockCount','Recent28WeekdayMean','Recent56WeekdayMean','LastObservedComparable','ComparableAgeDays','Recent7ClockMean','Recent28ClockMean','Lag7MinusLag14','Lag7To14Ratio']
CAL=['Weekday','Month','IsWeekend','FinalExamPeriod','CampusClosed','HolidayFlag']
STUDENTS=['WeeklyScheduledEnrollment','WeeklyActiveClasses','WeeklyClassesStarting','WeeklyClassesEnding','WeeklyDailyEnrollmentPeak']
EXTERNAL=['PriorDayTemperatureF','PriorDayPrecipitationIn','PriorDayMappedEventCount']
def jsonout(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,default=str))
def emit(s):print(s,flush=True)
def score(a,p):
 a=np.asarray(a,float);p=np.asarray(p,float);e=p-a;den=np.abs(a).sum()
 return dict(Observations=len(a),MAE=float(np.mean(abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),WAPE=float(100*abs(e).sum()/den) if den>0 else np.nan,WAPEDenominator=float(den),MeanError=float(e.mean()),MedianAbsoluteError=float(np.median(abs(e))),P90AbsoluteError=float(np.quantile(abs(e),.90)),ZeroActuals=int((a==0).sum()))
def split(d):return np.where(d.lt('2026-03-01'),'Train',np.where(d.lt('2026-05-01'),'Validation','Test'))
def init(root):
 for name in ['01_Approved_Data','03_Trained_Models/Saved_Model_Files','06_Model_Evaluation/Prediction_Ledger','06_Model_Evaluation','07_Feature_Importance_SHAP','09_Documentation']:(root/name).mkdir(parents=True,exist_ok=True)
def audit(root):
 S=BASE/'01_Approved_Data/Source_Files';h=pd.read_csv(BASE/'01_Approved_Data/Approved_ML_Flat_File.csv',parse_dates=['Date']);h=h.sort_values(['Location','Date','Hour']).reset_index(drop=True)
 checks=[]
 def ck(n,ok,detail='',severity='ERROR'):
  checks.append(dict(Check=n,Passed=bool(ok),Severity=severity,Detail=str(detail)))
  if severity=='ERROR' and not ok:raise AssertionError(n+':'+str(detail))
 ck('Hourly keys unique',not h.duplicated(['Location','Date','Hour']).any())
 ck('Dates and hours valid',h.Date.notna().all() and h.Hour.between(0,23).all() and h.Hour.mod(1).eq(0).all())
 ck('Location mapping',set(h.Location)=={"Einstein's",'Panda Express','Residential Dining'})
 ck('Chronological source split matches dates',np.array_equal(h.DataSplit,split(h.Date)))
 ck('Cutoff previous date',pd.to_datetime(h.ForecastCutoffDate).eq(h.Date-pd.Timedelta(days=1)).all())
 ck('Timestamp date/hour',pd.to_datetime(h.Timestamp).dt.normalize().eq(h.Date).all() and pd.to_datetime(h.Timestamp).dt.hour.eq(h.Hour).all())
 e=pd.read_csv(S/'Corrected_ETL/Fact_Hourly_Demand.csv').merge(pd.read_csv(S/'Corrected_ETL/Dim_Date.csv'),on='DateKey').merge(pd.read_csv(S/'Corrected_ETL/Dim_Time.csv'),on='TimeKey').merge(pd.read_csv(S/'Corrected_ETL/Dim_Location.csv'),on='LocationKey');e.Date=pd.to_datetime(e.Date)
 j=h.merge(e,on=['Location','Date','Hour'],how='outer',indicator=True,suffixes=('_ml','_etl'),validate='one_to_one')
 ck('Every ETL hourly target retained',j._merge.eq('both').all() and np.allclose(j.Target_HourlyUnits,j.TotalQuantity))
 ck('Net sales reconcile to ETL',np.allclose(j.NetSales_ml,j.NetSales_etl,atol=1e-6))
 ck('ETL validation passed',pd.read_csv(S/'Corrected_ETL/ETL_Validation_Report.csv').Status.eq('PASS').all())
 profile=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Class Weekday Hour')
 schedule=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Spring Class Schedule')
 ck('Profile weekday/hour unique',not profile.duplicated(['DayOfWeek','Hour']).any())
 ck('Enrollment equals seats minus available',np.allclose(schedule.EstimatedEnrolled,schedule.TotalSeats-schedule.AvailableSeats))
 ck('Profile nonnegative where present',profile[['ClassesActive','EstimatedStudentsOnCampus','ClassesStarting','ClassesEnding']].ge(0).all().all())
 prof=h.merge(profile,left_on=['Weekday','Hour'],right_on=['DayOfWeek','Hour'],how='left',suffixes=('_flat','_profile'))
 matched=prof.RegularClassDay.eq(1) & prof.ClassesActive_profile.notna()
 for col in ['ClassesActive','EstimatedStudentsOnCampus','ClassesStarting','ClassesEnding']:ck('Flat/profile '+col+' on regular matched hours',np.allclose(prof.loc[matched,col+'_flat'],prof.loc[matched,col+'_profile']))
 # Missing weekday/hour source rows are not zeroed in the rebuilt predictors.
 ck('Detailed schedule supports hour-by-hour enrollment reconstruction',schedule.StartTime.notna().any() and schedule.EndTime.notna().any(),'Current 588-row schedule has blank times; supplied weekly profile can be reconciled but not independently reconstructed from this workbook.',severity='LIMITATION')
 ck('Enrollment/event snapshot as-of timestamp verified',False,'No historical publication/capture timestamps supplied. Schedule-assisted results are conditional retrospective experiments, not certified deployment-safe.',severity='LIMITATION')
 ck('Operating calendar supplied',False,'No location opening/closure calendar; daily target is recorded-hour total, not verified complete operating-day demand.',severity='LIMITATION')
 b=pd.read_csv(S/'Baseline_Reference/input/ServeSmart_FinalFlatfile_Hourly_ML.csv',parse_dates=['Date']).sort_values(['Location','Date','Hour']).reset_index(drop=True)
 diffs=[]
 for c in h.columns:
  if pd.api.types.is_numeric_dtype(h[c]):
   bad=~np.isclose(pd.to_numeric(h[c]),pd.to_numeric(b[c]),equal_nan=True,atol=1e-7,rtol=1e-8)
  else:bad=h[c].fillna('<na>').astype(str).ne(b[c].fillna('<na>').astype(str)).to_numpy()
  if bad.any():diffs.append(dict(Column=c,DifferentRows=int(bad.sum()),MaximumAbsoluteDifference=float(np.nanmax(abs(h[c]-b[c]))) if pd.api.types.is_numeric_dtype(h[c]) else None))
 pd.DataFrame(diffs,columns=['Column','DifferentRows','MaximumAbsoluteDifference']).to_csv(root/'01_Approved_Data/Baseline_Input_Differences.csv',index=False)
 other=pd.read_csv(S/'ServeSmart_ML_Flat_File_Updated.csv',parse_dates=['Date']).sort_values(['Location','Date','Hour']).reset_index(drop=True)
 for col in other:
  a=h[col];b=other[col]
  ck('Other current ML column '+col,np.allclose(a,b,equal_nan=True) if pd.api.types.is_numeric_dtype(a) else a.fillna('<na>').astype(str).equals(b.fillna('<na>').astype(str)))
 pd.DataFrame(checks).to_csv(root/'01_Approved_Data/Data_Quality_Report.csv',index=False)
 profile.to_csv(root/'01_Approved_Data/Weekly_Class_Profile_Source.csv',index=False)
 schedule.to_csv(root/'01_Approved_Data/Class_Schedule_Enrollment_Source.csv',index=False)
 h[h.Target_HourlyUnits.lt(0)|h.Target_HourlyUnits.eq(0)].to_csv(root/'01_Approved_Data/Negative_And_Zero_Target_Rows.csv',index=False)
 jsonout(root/'01_Approved_Data/Data_Audit_Summary.json',dict(Rows=len(h),Units=float(h.Target_HourlyUnits.sum()),NetSales=float(h.NetSales.sum()),SplitCounts=h.DataSplit.value_counts().to_dict(),NegativeHours=int(h.Target_HourlyUnits.lt(0).sum()),ZeroHours=int(h.Target_HourlyUnits.eq(0).sum()),MissingRawProfileHours=int(prof.ClassesActive_profile.isna().sum()),SpringBreakRows=int(h.SpringBreak.sum()),InputPolicy='User-designated final 73-column ML snapshot, reconciled to corrected ETL; original 71-column alternative checked, original files preserved.'))
 return h,profile

def historical(frame,hourly):
 x=frame.sort_values(['Location','Date']+(['Hour'] if hourly else [])).reset_index(drop=True).copy()
 series={(loc,date,int(hour) if hourly else None):float(y) for loc,date,hour,y in zip(x.Location,x.Date,x.Hour if hourly else [0]*len(x),x.Actual)}
 daily=x.groupby(['Location','Date']).Actual.sum(min_count=1)
 features=[]
 for row in x.itertuples():
  date=row.Date;loc=row.Location;hour=int(row.Hour) if hourly else None;prior=x[(x.Location==loc)&(x.Date<date)]
  # Strictly prior calendar dates for every horizon-sensitive statistic.
  same=prior[(prior.Date.dt.dayofweek==date.dayofweek)&(prior.Hour.eq(hour) if hourly else True)]
  pday=daily.loc[loc];pday=pday[pday.index<date]
  v={f'Lag{k}Day':series.get((loc,date-pd.Timedelta(days=k),hour),np.nan) for k in [1,7,14]}
  v.update(HistoryWeekdayMean=same.Actual.mean(),HistoryWeekdayCount=same.Actual.count(),PrevDayTotal=daily.get((loc,date-pd.Timedelta(days=1)),np.nan))
  for k in [7,28]:
   last=pday[pday.index>=date-pd.Timedelta(days=k)];v[f'Prev{k}DayMean']=last.mean();v[f'Prev{k}DayCount']=last.count()
  clock=prior[prior.Hour.eq(hour)] if hourly else prior
  v.update(HistoryClockMean=clock.Actual.mean(),HistoryClockCount=clock.Actual.count(),Recent28WeekdayMean=same.loc[same.Date.ge(date-pd.Timedelta(days=28)),'Actual'].mean(),Recent56WeekdayMean=same.loc[same.Date.ge(date-pd.Timedelta(days=56)),'Actual'].mean(),LastObservedComparable=clock.Actual.iloc[-1] if len(clock) else np.nan,ComparableAgeDays=(date-clock.Date.iloc[-1]).days if len(clock) else np.nan,Recent7ClockMean=clock.loc[clock.Date.ge(date-pd.Timedelta(days=7)),'Actual'].mean(),Recent28ClockMean=clock.loc[clock.Date.ge(date-pd.Timedelta(days=28)),'Actual'].mean(),Lag7MinusLag14=v['Lag7Day']-v['Lag14Day'],Lag7To14Ratio=v['Lag7Day']/(abs(v['Lag14Day'])+1))
  features.append(v)
 return pd.concat([x,pd.DataFrame(features)],axis=1)

def prepare(root,h,profile):
 S=BASE/'01_Approved_Data/Source_Files'
 hourly=historical(h[['Location','Date','Hour','Target_HourlyUnits']].rename(columns={'Target_HourlyUnits':'Actual'}),True)
 daily=historical(h.groupby(['Location','Date'],as_index=False).agg(Actual=('Target_HourlyUnits','sum'),RecordedHours=('Hour','nunique')),False)
 calendar=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Academic Calendar');calendar.Date=pd.to_datetime(calendar.Date)
 cal=calendar[['Date','Final_Exam_Period','CampusClosedFlag','HolidayFlag']].rename(columns={'Final_Exam_Period':'FinalExamPeriod','CampusClosedFlag':'CampusClosed'});cal.FinalExamPeriod=cal.FinalExamPeriod.eq('Yes').astype(int)
 # Lagged source observations, not future realized weather/events.
 weather=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Weather Hourly');weather.Date=pd.to_datetime(weather.Date)
 w=weather.groupby('Date',as_index=False).agg(PriorDayTemperatureF=('TemperatureF','mean'),PriorDayPrecipitationIn=('PrecipitationIn',lambda x:x.sum(min_count=1)));w.Date+=pd.Timedelta(days=1)
 events=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Events Hourly');events.Date=pd.to_datetime(events.Date)
 ev=events.groupby('Date',as_index=False).agg(PriorDayMappedEventCount=('CampusEventCount',lambda x:x.sum(min_count=1)));ev.Date+=pd.Timedelta(days=1)
 peak=profile.groupby('DayOfWeek',as_index=False).agg(WeeklyDailyEnrollmentPeak=('EstimatedStudentsOnCampus','max'))
 pp=profile.rename(columns={'DayOfWeek':'Weekday','EstimatedStudentsOnCampus':'WeeklyScheduledEnrollment','ClassesActive':'WeeklyActiveClasses','ClassesStarting':'WeeklyClassesStarting','ClassesEnding':'WeeklyClassesEnding'})
 frames={}
 for level,x in [('Hourly',hourly),('Daily',daily)]:
  x['Weekday']=x.Date.dt.day_name();x['Month']=x.Date.dt.month;x['IsWeekend']=x.Date.dt.dayofweek.ge(5).astype(int);x['DataSplit']=split(x.Date);x['ForecastCutoff']=x.Date-pd.Timedelta(seconds=1)
  x=x.merge(cal,on='Date',how='left',validate='many_to_one').merge(w,on='Date',how='left',validate='many_to_one').merge(ev,on='Date',how='left',validate='many_to_one').merge(peak.rename(columns={'DayOfWeek':'Weekday'}),on='Weekday',how='left',validate='many_to_one')
  if level=='Hourly':x=x.merge(pp[['Weekday','Hour']+STUDENTS[:-1]],on=['Weekday','Hour'],how='left',validate='many_to_one')
  else:
   # Daily predictors use the full supplied weekly schedule, not future observed POS hours.
   pdaily=pp.groupby('Weekday',as_index=False).agg(WeeklyScheduledEnrollment=('WeeklyScheduledEnrollment','max'),WeeklyActiveClasses=('WeeklyActiveClasses','max'),WeeklyClassesStarting=('WeeklyClassesStarting','sum'),WeeklyClassesEnding=('WeeklyClassesEnding','sum'))
   x=x.merge(pdaily,on='Weekday',how='left',validate='many_to_one')
  x['ProfileMissing']=x.WeeklyScheduledEnrollment.isna().astype(int);frames[level]=x
  x.to_csv(root/f'01_Approved_Data/Prepared_{level}_Features_And_Actuals.csv',index=False)
 # Verify recomputed hourly lags against supplied values; rebuild means are same weekday/hour by design.
 compare=hourly.merge(h,on=['Location','Date','Hour'],validate='one_to_one')
 lagchecks=[]
 for k in [1,7,14]:
  ok=np.allclose(compare[f'Lag{k}Day'],compare[f'Lag{k}Day_SameHour'],equal_nan=True);lagchecks.append(dict(Check=f'Exact {k}-calendar-day hourly lag',Passed=ok));assert ok
 # Future-target perturbation invariance tests the entire feature construction path below separately.
 pd.DataFrame(lagchecks).to_csv(root/'01_Approved_Data/Feature_Validation_Report.csv',index=False)
 daily.to_csv(root/'01_Approved_Data/Derived_Daily_Actuals_And_History.csv',index=False)
 allow=set(HISTORY+CAL+STUDENTS+EXTERNAL+['Location','Hour','ProfileMissing'])
 banned=['SpringBreak','Spring_Break','Academic_Status','Academic_Phase','RegularClassDay','Regular_Class_Day','ScheduledStudents_DailyPeak','EstimatedStudentsOnCampus','ClassesActive','ClassesStarting','ClassesEnding','HasClass']
 auditrows=[]
 for c in h:
  reason='Excluded: rebuilt predictors use explicit allowlist; source identifiers, outcomes and keys are not selected.'
  if c in ['SpringBreak','Spring_Break','Academic_Status','Academic_Phase','RegularClassDay','Regular_Class_Day']:reason='Excluded: explicit Spring Break indicator or direct calendar-label/regular-class-day proxy.'
  if c in ['EstimatedStudentsOnCampus','ClassesActive','ClassesStarting','ClassesEnding','ScheduledStudents_DailyPeak','HasClass']:reason='Excluded: date-gated source may encode breaks/missing-as-zero; reconstruct ungated weekly schedule profile instead.'
  if c in ['TemperatureF','HumidityPct','PrecipitationIn','RainFlag','CloudCoverPct','WindSpeedKmh']:reason='Excluded: tomorrow actual weather unavailable at prior-day cutoff; use previous-day summaries only.'
  if 'Event' in c:reason='Excluded contemporaneous event value/identifier: publication as-of not verified; use lagged mapped-event intensity only.'
  if c in ['Location','Hour','Weekday','Month','IsWeekend','FinalExamPeriod','CampusClosed','HolidayFlag']:reason='Allowed calendar/location identity; calendar fields cross-checked; no break-label encoding.'
  auditrows.append(dict(SourceColumn=c,Use='Allowed' if c in allow else 'Excluded',Reason=reason))
 pd.DataFrame(auditrows).to_csv(root/'01_Approved_Data/Predictor_Source_Column_Audit.csv',index=False)
 jsonout(root/'01_Approved_Data/Approved_Predictor_List.json',{'history':HISTORY,'calendar':CAL,'schedule':STUDENTS+['ProfileMissing'],'past_weather_events':EXTERNAL,'excluded_break_and_direct_proxy_columns':banned,'availability':'Schedule counts are conditional on the supplied weekly enrollment snapshot being available at cutoff; no historical publication evidence was supplied. All demand/weather/event histories are strictly earlier calendar dates. No statement of deployment readiness.'})
 availability=[]
 for c in sorted(allow):availability.append(dict(Predictor=c,Availability='Conditional: supplied published weekly schedule snapshot; actual historical publication timestamp unknown' if c in STUDENTS+['ProfileMissing'] else 'Deterministic date/location/calendar' if c in CAL+['Location','Hour'] else 'Strictly previous calendar dates; assumes historical records available by end of day',MissingTreatment='Training-only median plus missing indicator; unknown profile rows are not zero-filled'))
 pd.DataFrame(availability).to_csv(root/'01_Approved_Data/Predictor_Availability_Audit.csv',index=False)
 return frames

def features(level,kind):
 base=['Location']+(['Hour'] if level=='Hourly' else [])+HISTORY
 if kind in ['HistoryCalendar','Full']:base+=CAL
 if kind in ['HistoryEnrollment','Full']:base+=STUDENTS+['ProfileMissing']
 if kind in ['HistoryWeatherEvents','Full']:base+=EXTERNAL
 return list(dict.fromkeys(base))
def pipe(model,cols,train):
 cat=[c for c in cols if c in ['Location','Weekday']];num=[c for c in cols if c not in cat]
 prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)],sparse_threshold=0)
 return Pipeline([('preprocess',prep),('model',model)])
def estimator(name,config):
 # Package B's two documented settings per tree family, re-evaluated on SAFE features.
 if config>=3 and name in ['Random Forest','XGBoost','LightGBM']:
  conf=config-3
  if name=='Random Forest':return RandomForestRegressor(n_estimators=[200,300][conf],max_depth=[8,None][conf],max_features=.8,min_samples_leaf=2,n_jobs=2,random_state=42)
  if name=='XGBoost':return XGBRegressor(n_estimators=[200,300][conf],max_depth=[4,3][conf],learning_rate=.05,subsample=.9,colsample_bytree=.9,objective='reg:squarederror',n_jobs=2,random_state=42)
  if name=='LightGBM':return LGBMRegressor(n_estimators=[200,300][conf],num_leaves=[15,31][conf],learning_rate=[.05,.03][conf],subsample=.9,colsample_bytree=.9,n_jobs=2,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
 if name=='Ridge':return Ridge(alpha=[1.,10.,100.][config])
 if name=='Random Forest':return RandomForestRegressor(n_estimators=180,max_depth=[None,8,5][config],min_samples_leaf=[2,4,8][config],max_features=[1.,.8,1.][config],n_jobs=2,random_state=SEED)
 if name=='XGBoost':return XGBRegressor(n_estimators=[180,300,180][config],max_depth=[2,3,4][config],learning_rate=[.03,.03,.05][config],min_child_weight=4,subsample=.9,colsample_bytree=.9,reg_lambda=5,objective='reg:squarederror',n_jobs=2,random_state=SEED)
 if name=='LightGBM':return LGBMRegressor(n_estimators=[180,300,180][config],num_leaves=[7,15,7][config],min_child_samples=[10,15,25][config],learning_rate=.03,reg_lambda=5,verbosity=-1,n_jobs=2,random_state=SEED,deterministic=True,force_col_wise=True)
 raise ValueError(name)
def prophet_fit(train,cols,level,config):
 models={};nums=[c for c in cols if c not in ['Location','Weekday','Hour']]
 for loc,t in train.groupby('Location'):
  t=t.copy();med=t[nums].median().fillna(0);active=[c for c in nums if t[c].fillna(med[c]).nunique()>1]
  t['ds']=t.Date+pd.to_timedelta(t.Hour,unit='h') if level=='Hourly' else t.Date;t['y']=t.Actual
  t[nums]=t[nums].fillna(med)
  m=Prophet(changepoint_prior_scale=[.01,.1][config],seasonality_prior_scale=1.,weekly_seasonality=3,daily_seasonality=level=='Hourly',yearly_seasonality=False,uncertainty_samples=0)
  for c in active:m.add_regressor(c,prior_scale=.5)
  with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(t[['ds','y']+active],seed=SEED)
  models[loc]=(m,med,active)
 return models

def prophet_predict(models,test,level):
 pred=pd.Series(index=test.index,dtype=float)
 for loc,t in test.groupby('Location'):
  m,med,active=models[loc];u=t.copy();u['ds']=u.Date+pd.to_timedelta(u.Hour,unit='h') if level=='Hourly' else u.Date;u[active]=u[active].fillna(med)
  pred.loc[t.index]=m.predict(u[['ds']+active]).yhat.to_numpy()
 return np.maximum(0,pred.to_numpy())

def save_prophet(models,path):
 obj={loc:dict(model=model_to_json(m),median=med.to_dict(),active=active) for loc,(m,med,active) in models.items()};jsonout(path,obj)
def load_prophet(path):
 return {loc:(model_from_json(v['model']),pd.Series(v['median']),v['active']) for loc,v in json.loads(path.read_text()).items()}
def record(frame,p,level,name,kind,cohort):
 cols=['Location','Date','Hour','Weekday','Actual','ForecastCutoff'] if level=='Hourly' else ['Location','Date','Weekday','Actual','RecordedHours','ForecastCutoff']
 z=frame[cols].copy();z['Prediction']=p;z['ForecastLevel']=level;z['Model']=name;z['FeatureSet']=kind;z['Cohort']=cohort;z['AbsoluteError']=abs(z.Prediction-z.Actual);z['Error']=z.Prediction-z.Actual
 return z

def train_validate(root,frames):
 candidates=[];vp=[];chosen={};status=[]
 for level,x in frames.items():
  tr=x[(x.DataSplit=='Train')&x.Actual.ge(0)];va=x[x.DataSplit=='Validation'];common=va.Lag7Day.notna()&va.HistoryWeekdayMean.notna();emit(f'{level}: Train={len(tr)}, validation={len(va)}, common validation={common.sum()}')
  best={}
  for name,col in [('Seasonal Naive','Lag7Day'),('Historical Average','HistoryWeekdayMean')]:
   pred=va[col].clip(lower=0).to_numpy();v=record(va,pred,level,name,'HistoricalOnly','Validation');vp.append(v)
   m=score(va.loc[common,'Actual'],pred[common]);candidates.append(dict(ForecastLevel=level,Model=name,FeatureSet='HistoricalOnly',Config=0,**m));best[name]=candidates[-1]
  for name in METHODS[2:]:
   configs=range(2) if name=='Prophet' else range(3) if name=='Ridge' else range(5)
   # Prophet uses the same feature ablations; no missing-hour zero padding.
   for kind in SETS:
    cols=features(level,kind)
    for conf in configs:
     start=time.monotonic()
     try:
      if name=='Prophet':model=prophet_fit(tr,cols,level,conf);pred=prophet_predict(model,va,level)
      else:model=pipe(estimator(name,conf),cols,tr);model.fit(tr[cols],tr.Actual);pred=np.maximum(0,model.predict(va[cols]))
      m=score(va.loc[common,'Actual'],pred[common]);row=dict(ForecastLevel=level,Model=name,FeatureSet=kind,Config=conf,Seconds=round(time.monotonic()-start,3),**m);candidates.append(row)
      if name not in best or (m['MAE'],m['RMSE'])<(best[name]['MAE'],best[name]['RMSE']):best[name]=row
     except Exception as exc:
      status.append(dict(ForecastLevel=level,Model=name,FeatureSet=kind,Config=conf,Status='Failed',Reason=repr(exc)));emit(f'FAILED {level} {name} {kind}: {exc}')
    emit(f'Validated {level}: {name} / {kind}')
   if name in best:
    b=best[name];cols=features(level,b['FeatureSet'])
    if name=='Prophet':model=prophet_fit(tr,cols,level,b['Config']);p=prophet_predict(model,va,level)
    else:model=pipe(estimator(name,b['Config']),cols,tr);model.fit(tr[cols],tr.Actual);p=np.maximum(0,model.predict(va[cols]))
    vp.append(record(va,p,level,name,b['FeatureSet'],'Validation'))
  ranked=sorted(best.values(),key=lambda z:(z['MAE'],z['RMSE'],METHODS.index(z['Model'])))
  for rank,b in enumerate(ranked,1):b['ValidationRank']=rank;chosen[(level,b['Model'])]=b
  jsonout(root/f'03_Trained_Models/{level}_Selection_Locked_Before_Test.json',dict(Winner=ranked[0],AllMethods=ranked,SelectionRule='Lowest common-validation MAE; RMSE tie break. Fixed in this file before any May scoring.'))
  status.extend(dict(ForecastLevel=level,Model=m,Status='Validated' if m in best else 'Unavailable',Reason='') for m in METHODS)
 pd.DataFrame(candidates).to_csv(root/'06_Model_Evaluation/All_Validation_Experiments.csv',index=False)
 pd.concat(vp,ignore_index=True).to_csv(root/'06_Model_Evaluation/Prediction_Ledger/Validation_Predictions.csv',index=False)
 pd.DataFrame(status).to_csv(root/'06_Model_Evaluation/Model_Execution_Status.csv',index=False)
 return chosen

def evaluate(root,frames,chosen):
 predictions=[];registry=[];loadedchecks=[];models={}
 # No hyperparameters, feature sets or winners may change beyond this line.
 for (level,name),choice in chosen.items():
  x=frames[level];tr=x[(x.DataSplit!='Test')&x.Actual.ge(0)];te=x[x.DataSplit=='Test'];kind=choice['FeatureSet'];cols=features(level,kind);slug=level.lower()+'_'+name.lower().replace(' ','_');path=root/'03_Trained_Models/Saved_Model_Files'/slug
  if name in METHODS[:2]:
   col='Lag7Day' if name=='Seasonal Naive' else 'HistoryWeekdayMean';pred=te[col].clip(lower=0).to_numpy();path=path.with_suffix('.json');jsonout(path,dict(Model=name,PredictionFeature=col,Policy='Exact previous calendar week / expanding same weekday and hour; never fill missing with zero; clip predictions at zero.'))
   reloaded=te[json.loads(path.read_text())['PredictionFeature']].clip(lower=0).to_numpy()
  elif name=='Prophet':
   model=prophet_fit(tr,cols,level,choice['Config']);pred=prophet_predict(model,te,level);path=path.with_suffix('.json');save_prophet(model,path);reloaded=prophet_predict(load_prophet(path),te,level);models[(level,name)]=model
  else:
   model=pipe(estimator(name,choice['Config']),cols,tr);model.fit(tr[cols],tr.Actual);pred=np.maximum(0,model.predict(te[cols]));path=path.with_suffix('.joblib');joblib.dump(model,path);reloaded=np.maximum(0,joblib.load(path).predict(te[cols]));models[(level,name)]=model
  assert np.allclose(pred,reloaded,equal_nan=True,atol=1e-8);loadedchecks.append(dict(ForecastLevel=level,Model=name,ReloadPredictionMatch=True))
  predictions.append(record(te,pred,level,name,kind,'Test'))
  registry.append(dict(ForecastLevel=level,Model=name,FeatureSet=kind,Features=cols,ModelFile=str(path.relative_to(root)),ValidationRank=choice['ValidationRank'],Selected=choice['ValidationRank']==1,TrainingCutoff='2026-04-30',PredictionPolicy='One calendar day ahead with rolling historical features; fixed coefficients after April30. Conditional on supplied schedule snapshot for enrollment variants.'))
 emit('All test predictions generated; model selection remains locked.')
 p=pd.concat(predictions,ignore_index=True);p['CommonEvaluationRow']=False
 rows=[];breakdowns=[]
 for level,lp in p.groupby('ForecastLevel'):
  keys=['Location','Date']+(['Hour'] if level=='Hourly' else [])
  counts=lp[lp.Prediction.notna()].groupby(keys).Model.nunique();eligible=counts[counts==lp.Model.nunique()].reset_index()[keys]
  for name,g in lp.groupby('Model'):
   common=g.merge(eligible,on=keys,how='inner');mask=pd.MultiIndex.from_frame(g[keys]).isin(pd.MultiIndex.from_frame(eligible));p.loc[g.index,'CommonEvaluationRow']=mask
   b=chosen[(level,name)];rows.append(dict(ForecastLevel=level,Model=name,FeatureSet=b['FeatureSet'],ValidationRank=b['ValidationRank'],ValidationMAE=b['MAE'],SelectedByValidation=b['ValidationRank']==1,Evaluation='Common May observations across all seven methods',**score(common.Actual,common.Prediction)))
   for gt,cols2 in [('Location',['Location']),('Weekday',['Weekday']),('LocationWeekday',['Location','Weekday'])]+([('Hour',['Hour']),('LocationHour',['Location','Hour'])] if level=='Hourly' else []):
    for vals,z in common.groupby(cols2,dropna=False):
     vals=vals if isinstance(vals,tuple) else (vals,);breakdowns.append(dict(ForecastLevel=level,Model=name,GroupType=gt,**dict(zip(cols2,vals)),**score(z.Actual,z.Prediction)))
 p.to_csv(root/'06_Model_Evaluation/Prediction_Ledger/All_Test_Predictions.csv',index=False)
 for level in frames:
  p[p.ForecastLevel.eq(level)].to_csv(root/f'06_Model_Evaluation/Prediction_Ledger/{level}_All_Model_Predictions.csv',index=False)
  win=next(k[1] for k,v in chosen.items() if k[0]==level and v['ValidationRank']==1)
  p[p.ForecastLevel.eq(level)&p.Model.eq(win)].to_csv(root/f'06_Model_Evaluation/Prediction_Ledger/{level}_Predictions.csv',index=False)
 comparison=pd.DataFrame(rows).sort_values(['ForecastLevel','ValidationRank']);comparison.to_csv(root/'06_Model_Evaluation/Overall_Model_Comparison.csv',index=False)
 detail=pd.DataFrame(breakdowns);detail.to_csv(root/'06_Model_Evaluation/All_Group_Performance.csv',index=False)
 for group,file in [('Location','Location_Wise_Performance.csv'),('Weekday','Weekday_Wise_Performance.csv'),('Hour','Hour_Wise_Performance.csv')]:detail[detail.GroupType.eq(group)].to_csv(root/'06_Model_Evaluation'/file,index=False)
 detail[detail.ForecastLevel.eq('Daily')].to_csv(root/'06_Model_Evaluation/Daily_Performance.csv',index=False)
 detail[detail.GroupType.isin(['LocationWeekday','LocationHour'])].to_csv(root/'06_Model_Evaluation/Location_Time_Performance.csv',index=False)
 pd.DataFrame(loadedchecks).to_csv(root/'06_Model_Evaluation/Model_Reload_Checks.csv',index=False);jsonout(root/'03_Trained_Models/Model_Registry.json',registry)
 # Compare aggregate-of-observed-hour forecasts with independently trained daily models.
 ag=p[p.ForecastLevel.eq('Hourly')].groupby(['Model','Location','Date'],as_index=False).agg(HourlyPredictionSum=('Prediction',lambda x:x.sum(min_count=1)),PredictedObservedHours=('Prediction','count'))
 d=frames['Daily'];te=d[d.DataSplit.eq('Test')];ag=ag.merge(te[['Location','Date','Actual','RecordedHours']],on=['Location','Date'],validate='many_to_one');ag['CompleteObservedHourSet']=ag.PredictedObservedHours.eq(ag.RecordedHours)
 ag.to_csv(root/'06_Model_Evaluation/Prediction_Ledger/Daily_From_Observed_Hourly_Predictions.csv',index=False)
 ar=[]
 common_daily=pd.MultiIndex.from_frame(p[(p.ForecastLevel=='Daily')&p.CommonEvaluationRow][['Location','Date']].drop_duplicates())
 for name,g in ag.groupby('Model'):
  good=g.CompleteObservedHourSet & pd.MultiIndex.from_frame(g[['Location','Date']]).isin(common_daily)
  if good.any():ar.append(dict(Model=name,Scope='Sum over recorded test hours only; NOT a complete operating-day forecast',**score(g.loc[good,'Actual'],g.loc[good,'HourlyPredictionSum'])))
 pd.DataFrame(ar).to_csv(root/'06_Model_Evaluation/Hourly_To_Daily_Aggregation_Comparison.csv',index=False)
 return p,comparison,detail,registry,models

def explain(root,frames,registry,models):
 from sklearn.inspection import permutation_importance
 files=[]
 for reg in registry:
  best_tree=min((r['ValidationRank'] for r in registry if r['ForecastLevel']==reg['ForecastLevel'] and r['Model'] in ['Random Forest','XGBoost','LightGBM']),default=999)
  if not reg['Selected'] and not (reg['Model'] in ['Random Forest','XGBoost','LightGBM'] and reg['ValidationRank']==best_tree):continue
  level=reg['ForecastLevel'];name=reg['Model'];x=frames[level];va=x[x.DataSplit.eq('Validation')&x.Lag7Day.notna()&x.HistoryWeekdayMean.notna()];tr=x[x.DataSplit.eq('Train')&x.Actual.ge(0)];cols=reg['Features']
  if name in ['Seasonal Naive','Historical Average','Prophet']:
   # Prophet decomposition and baseline identity remain exact and interpretable.
   if name in ['Seasonal Naive','Historical Average']:
    selected=pd.read_csv(root/f'06_Model_Evaluation/Prediction_Ledger/{level}_Predictions.csv',parse_dates=['Date'])
    examples=selected.groupby('Location',sort=True).head(1).copy()
    history=x[['Location','Date']+(['Hour'] if level=='Hourly' else [])+['HistoryWeekdayCount','HistoryWeekdayMean','Lag7Day']]
    examples=examples.merge(history,on=['Location','Date']+(['Hour'] if level=='Hourly' else []),validate='one_to_one')
    examples['Explanation']='Equal-weight mean of earlier same-weekday observations'+(' at the same clock hour' if level=='Hourly' else '') if name=='Historical Average' else 'Observed demand exactly seven calendar days earlier'
    examples.to_csv(root/f'07_Feature_Importance_SHAP/{level}_Selected_Baseline_Examples.csv',index=False)
    ax=examples.plot.bar(x='Location',y=['Actual','Prediction'],figsize=(8,4),rot=0);ax.set_ylabel('Recorded POS units');ax.set_title(f'{level}: selected {name}; first May observation per location');plt.tight_layout();plt.savefig(root/f'07_Feature_Importance_SHAP/{level}_Selected_Baseline_Explanation.png',dpi=160);plt.close()
    files.append(dict(Level=level,Method='Exact historical contributors / baseline formula',Status='Generated',Note='Selected model is a baseline; tree importance files, if present, describe the best validation-ranked advanced tree, not the selected baseline.'))
   if name=='Prophet':
    for loc,(model,med,active) in models[(level,name)].items():
     t=x[(x.Location==loc)&x.DataSplit.eq('Test')].copy();t['ds']=t.Date+pd.to_timedelta(t.Hour,unit='h') if level=='Hourly' else t.Date;t[active]=t[active].fillna(med);fc=model.predict(t[['ds']+active]);model.plot_components(fc);plt.savefig(root/f'07_Feature_Importance_SHAP/{level}_{loc.replace(" ","_")}_Prophet_Components.png',bbox_inches='tight');plt.close('all')
   continue
  # Importance computed on validation with a TRAIN-ONLY model; no May-based selection.
  choice=json.loads((root/f'03_Trained_Models/{level}_Selection_Locked_Before_Test.json').read_text());ch=next(c for c in choice['AllMethods'] if c['Model']==name)
  model=pipe(estimator(name,ch['Config']),cols,tr);model.fit(tr[cols],tr.Actual)
  imp=permutation_importance(model,va[cols],va.Actual,n_repeats=5,random_state=SEED,scoring='neg_mean_absolute_error',n_jobs=1)
  table=pd.DataFrame(dict(Feature=cols,ValidationMAEIncrease=imp.importances_mean,Std=imp.importances_std)).sort_values('ValidationMAEIncrease',ascending=False);table.to_csv(root/f'07_Feature_Importance_SHAP/{level}_Permutation_Importance.csv',index=False)
  ax=table.head(15).iloc[::-1].plot.barh(x='Feature',y='ValidationMAEIncrease',legend=False,figsize=(9,6));ax.set_xlabel('Validation MAE increase when shuffled');ax.set_title(f'{level}: {name} feature importance (association only)');plt.tight_layout();plt.savefig(root/f'07_Feature_Importance_SHAP/{level}_Feature_Importance.png',dpi=160);plt.close()
  final=models[(level,name)];te=x[x.DataSplit.eq('Test')].head(120)
  if name in ['Random Forest','XGBoost','LightGBM']:
   try:
    import shap
    trans=final.named_steps['preprocess'];a=trans.transform(te[cols]);names=trans.get_feature_names_out();explainer=shap.TreeExplainer(final.named_steps['model']);values=explainer(a,check_additivity=False)
    shap.summary_plot(values.values,a,feature_names=names,show=False,max_display=15);plt.title(f'{level}: {name} SHAP; model-scale output');plt.tight_layout();plt.savefig(root/f'07_Feature_Importance_SHAP/{level}_SHAP_Analysis.png',dpi=160,bbox_inches='tight');plt.close()
    np.savez_compressed(root/f'07_Feature_Importance_SHAP/{level}_SHAP_Values.npz',values=values.values,base_values=values.base_values,data=a,feature_names=names.astype(str))
    for i in [0,min(1,len(te)-1),min(2,len(te)-1)]:
     order=np.argsort(abs(values.values[i]))[::-1][:8];pd.DataFrame({'Feature':names[order],'SHAPContribution':values.values[i][order],'TransformedFeatureValue':a[i,order]}).to_csv(root/f'07_Feature_Importance_SHAP/{level}_Example_{i+1}_Contributions.csv',index=False)
    files.append(dict(Level=level,Method='SHAP TreeExplainer',Status='Generated',Note='Additive raw tree output; public predictions additionally clipped at zero. Examples match the first three test records for the explained advanced tree; this tree is not the selected baseline.'))
   except Exception as exc:files.append(dict(Level=level,Method='SHAP TreeExplainer',Status='Unavailable',Reason=repr(exc)))
 pd.DataFrame(files).to_csv(root/'07_Feature_Importance_SHAP/Explainability_Status.csv',index=False)

def validation_daily_approaches(root,frames):
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

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-run',default='',help='Optional separate run folder under 09_Reproduction_Runs; no original inputs changed');parser.add_argument('--skip-explainability',action='store_true');args=parser.parse_args()
 root=BASE/'09_Reproduction_Runs'/args.output_run if args.output_run else BASE;init(root)
 start=time.monotonic();h,profile=audit(root);frames=prepare(root,h,profile);chosen=train_validate(root,frames);validation_daily_approaches(root,frames);pred,comparison,detail,registry,models=evaluate(root,frames,chosen)
 if not args.skip_explainability:explain(root,frames,registry,models)
 jsonout(root/'09_Documentation/Run_Manifest.json',dict(Seed=SEED,Seconds=time.monotonic()-start,Python=sys.version,Packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','xgboost','lightgbm','prophet','joblib','matplotlib']},InputSHA256=hashlib.sha256((BASE/'01_Approved_Data/Approved_ML_Flat_File.csv').read_bytes()).hexdigest(),Split='Train Jan-Feb2026, Validation Mar-Apr2026, Test May2026',TestSelectionPolicy='Winners locked before May prediction generation; May has been used in earlier project development, so this is a historical benchmark, not a prospectively untouched dataset.',DailyTarget='Recorded location-day net POS units; incomplete operating-hour coverage not equated to full-day demand',ScheduleAvailability='Conditional; historical schedule snapshot timing unverified'))
 publish_level_outputs(root)
 emit(comparison[['ForecastLevel','Model','FeatureSet','Observations','MAE','RMSE','WAPE','ValidationRank']].to_string(index=False));emit('Pipeline finished.')
if __name__=='__main__':main()
