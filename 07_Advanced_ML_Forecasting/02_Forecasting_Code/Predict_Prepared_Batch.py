"""Inference from explicitly prepared as-of features; does not invent future inputs.
Example: python 02_Forecasting_Code/Predict_Prepared_Batch.py --level Hourly --input input.csv --output new_predictions.csv
See 09_Documentation/Website_Handoff.md before any live use.
"""
import os,sys,importlib.util
from pathlib import Path
if sys.platform=='darwin' and not os.environ.get('SERVESMART_RUNTIME_READY'):
 spec=importlib.util.find_spec('sklearn');env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(spec.origin).parent/'.dylibs');env['SERVESMART_RUNTIME_READY']='1';os.execve(sys.executable,[sys.executable,*sys.argv],env)
import argparse,json,joblib,pandas as pd,numpy as np
from Forecasting_Pipeline import load_prophet,prophet_predict
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--level',choices=['Hourly','Daily'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',default='',help='Default is locked validation winner');a=p.parse_args()
if a.output.exists():raise SystemExit('Choose a new output filename; existing files are never overwritten.')
registry=json.loads((R/'03_Trained_Models/Model_Registry.json').read_text());reg=next(r for r in registry if r['ForecastLevel']==a.level and (r['Model']==a.model if a.model else r['Selected']))
x=pd.read_csv(a.input,parse_dates=['Date']);keys=['Location','Date']+(['Hour'] if a.level=='Hourly' else [])
assert not x.duplicated(keys).any(),'Duplicate prediction keys'
assert set(x.Location)<=set(["Einstein's",'Panda Express','Residential Dining']),'Unknown location'
if a.level=='Hourly':assert x.Hour.between(0,23).all() and x.Hour.mod(1).eq(0).all()
assert 'ForecastCutoff' in x,'Provide an explicit as-of cutoff for every prediction'
x.ForecastCutoff=pd.to_datetime(x.ForecastCutoff);assert x.ForecastCutoff.eq(x.Date-pd.Timedelta(seconds=1)).all(),'Next-day cutoffs must be 23:59:59 on the previous day'
name=reg['Model'];path=R/reg['ModelFile']
if name in ['Historical Average','Seasonal Naive']:
 meta=json.loads(path.read_text());pred=x[meta['PredictionFeature']].clip(lower=0).to_numpy()
elif name=='Prophet':pred=prophet_predict(load_prophet(path),x,a.level)
else:pred=np.maximum(0,joblib.load(path).predict(x[reg['Features']]))
out=x[keys+['ForecastCutoff']].copy();out['Prediction']=pred;out['Model']=name;out['InputStatus']='Caller-supplied as-of features; availability/operating schedule must be validated by caller';a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False);print(a.output)
