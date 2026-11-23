"""Additional forecast-cutoff stress checks and reproducible explanations."""
import os,sys,importlib.util
from pathlib import Path
if sys.platform=='darwin' and not os.environ.get('SERVESMART_RUNTIME_READY'):
 spec=importlib.util.find_spec('sklearn');env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(spec.origin).parent/'.dylibs');env['SERVESMART_RUNTIME_READY']='1';os.execve(sys.executable,[sys.executable,*sys.argv],env)
import pandas as pd,numpy as np,joblib
import Forecasting_Pipeline as P
R=P.BASE
frames={l:pd.read_csv(R/f'01_Approved_Data/Prepared_{l}_Features_And_Actuals.csv',parse_dates=['Date']) for l in ['Hourly','Daily']}
checks=[]
for level,x in frames.items():
 cols=['Location','Date','Actual']+(['Hour'] if level=='Hourly' else [])
 a=x[cols].copy();b=a.copy();cut=pd.Timestamp('2026-04-15');b.loc[b.Date.ge(cut),'Actual']+=1000000
 original=P.historical(a,level=='Hourly');changed=P.historical(b,level=='Hourly')
 before=original.Date.le(cut);assert np.allclose(original.loc[before,P.HISTORY],changed.loc[before,P.HISTORY],equal_nan=True)
 # Daily future coverage must not determine today's schedule predictor.
 assert 'RecordedHours' not in P.features(level,'Full')
 checks.append(dict(Level=level,FutureTargetPerturbationInvariant=True,IncludesCutoffDay=True,NoRecordedHoursPredictor=True,Test='Add one million units to all targets on/after April15; every history predictor on/before that day remains identical.'))
P.jsonout(R/'06_Model_Evaluation/Leakage_Stress_Checks.json',checks)
registry=__import__('json').loads((R/'03_Trained_Models/Model_Registry.json').read_text());models={}
for reg in registry:
 if reg['Model'] in ['Ridge','Random Forest','XGBoost','LightGBM']:models[(reg['ForecastLevel'],reg['Model'])]=joblib.load(R/reg['ModelFile'])
 elif reg['Model']=='Prophet':models[(reg['ForecastLevel'],reg['Model'])]=P.load_prophet(R/reg['ModelFile'])
P.explain(R,frames,registry,models)
print('PASS: future-target perturbation invariance and explanation generation.')
