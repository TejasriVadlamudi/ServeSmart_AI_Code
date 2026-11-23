"""Check context joins and original-source preservation independently of modeling."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];S=R/'01_Approved_Data/Source_Files';h=pd.read_csv(R/'01_Approved_Data/Approved_ML_Flat_File.csv',parse_dates=['Date']);rows=[]
def ck(name,ok,note=''):
 rows.append(dict(Check=name,Passed=bool(ok),Detail=str(note)));assert ok,name
w=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Weather Hourly');w.Date=pd.to_datetime(w.Date);ck('Weather source date/hour unique',not w.duplicated(['Date','Hour']).any())
j=h.merge(w,on=['Date','Hour'],how='left',suffixes=('_ml','_source'),validate='many_to_one')
for c in ['TemperatureF','HumidityPct','PrecipitationIn','RainFlag','CloudCoverPct','WindSpeedKmh']:ck('Aligned weather '+c,np.allclose(j[c+'_ml'],j[c+'_source'],equal_nan=True))
e=pd.read_excel(S/'01-ServeSmart_Master-Dataset-Copy.xlsx',sheet_name='Events Hourly');e.Date=pd.to_datetime(e.Date);ck('Event source date/hour unique',not e.duplicated(['Date','Hour']).any());j=h.merge(e,on=['Date','Hour'],how='left',suffixes=('_ml','_source'),validate='many_to_one')
for c in ['CampusEventCount','EventOccurrenceCount','EventFoodCount','AllDayEventCount','EventTypeCount']:ck('Aligned supplied event '+c,np.allclose(j[c+'_ml'],j[c+'_source'].fillna(0)))
# Prove each lagged external feature equals the previous calendar day's source summary.
w=w.groupby('Date').agg(TemperatureF=('TemperatureF','mean'),PrecipitationIn=('PrecipitationIn',lambda v:v.sum(min_count=1)))
e=e.groupby('Date').CampusEventCount.sum(min_count=1)
for level in ['Hourly','Daily']:
 x=pd.read_csv(R/f'01_Approved_Data/Prepared_{level}_Features_And_Actuals.csv',parse_dates=['Date'])
 for r in x.itertuples():
  prev=r.Date-pd.Timedelta(days=1)
  for c,s,v in [('PriorDayTemperatureF',w.TemperatureF,r.PriorDayTemperatureF),('PriorDayPrecipitationIn',w.PrecipitationIn,r.PriorDayPrecipitationIn),('PriorDayMappedEventCount',e,r.PriorDayMappedEventCount)]:assert np.isclose(v,s.get(prev,np.nan),equal_nan=True)
 ck(level+' prior-day external features align',True)
 for key,group in x.groupby('Date'):
  # No use of observed outlet hours in schedule generation.
  assert group.WeeklyDailyEnrollmentPeak.nunique(dropna=False)==1
 ck(level+' daily enrollment peak independent of outlet hours',True)
manifest=json.loads((R/'01_Approved_Data/Input_Manifest.json').read_text())
for entry in manifest['sources']:
 p=Path(entry['path']);ck('Original source unchanged: '+p.name,p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'])
# Baseline source folder was copied byte-for-byte and never run/edited in place.
original=Path.home()/'Desktop/06_Baseline_Forecasting';copy=S/'Baseline_Reference';count=0
for f in copy.rglob('*'):
 if f.is_file():assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256((original/f.relative_to(copy)).read_bytes()).digest();count+=1
ck('Original baseline folder unchanged',True,count)
pd.DataFrame(rows).to_csv(R/'06_Model_Evaluation/Context_And_Original_File_Verification.csv',index=False);print('PASS:',len(rows),'context/source preservation checks.')
