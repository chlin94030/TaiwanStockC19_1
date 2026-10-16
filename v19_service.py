"""Persistent research jobs and display contracts for V19.

The app owns its worker, not the assistant. No orders are placed. Successful
reports survive a failed refresh; mobile reconnects do not resubmit the job.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import hashlib
import os
import threading
import time
import numpy as np
import pandas as pd
from v19_config import Settings, TZ, FACTORS, TECH, prior
from v19_data import (download_history, import_legacy_sqlite, read_json, atomic_json,
                      atomic_csv, clean_prices, SourceError)
from v19_features import build_panel, current_candidates
from v19_backtest import analyze, add_outcomes, save_analysis

_LOCK=threading.RLock()
_JOBS={}


def paths(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    return {'root':root,'state':root/'v19_job.json','last':root/'last_success.json','runroot':root/'runs'}


def job_status(root):
    p=paths(root);key=str(p['root'].resolve());obj=read_json(p['state'])
    with _LOCK:alive=bool(_JOBS.get(key) and _JOBS[key].is_alive())
    if obj.get('state')=='RUNNING' and not alive:
        pid=obj.get('pid')
        if pid==os.getpid() or time.time()-float(obj.get('updated_epoch',0))>120:
            obj=dict(obj,state='INTERRUPTED',message='Worker stopped; cached prices retained. Resume the job.')
    return obj


def read_last_report(root):
    p=paths(root);last=read_json(p['last'])
    target=last.get('directory')
    if target:
        path=Path(target).resolve()
        if path.is_relative_to(p['runroot'].resolve()):return read_json(path/'v19_results.json')
    return {}


def load_inputs(root,cfg=Settings()):
    root=Path(root);pricepath=root/'history_prices.csv.gz';evpath=root/'historical_evidence.csv.gz'
    prices=pd.read_csv(pricepath) if pricepath.exists() else pd.DataFrame()
    if prices.empty:
        for candidate in (root.parent/'daily_prices.sqlite',root/'daily_prices.sqlite'):
            if candidate.exists():
                prices=import_legacy_sqlite(candidate,cfg.start(),cfg.end())
                if not prices.empty:break
    evidence=pd.read_csv(evpath) if evpath.exists() else pd.DataFrame()
    return prices,evidence


def material_for_display(root,cfg=Settings()):
    prices,evidence=load_inputs(root,cfg)
    if prices.empty:return pd.DataFrame(),{}
    panel=build_panel(prices,evidence,cfg);panel,_=add_outcomes(panel,cfg)
    return panel,read_last_report(root)


def start_job(root,cfg=Settings(),mode='download_validate'):
    if mode not in ('download_validate','cache_validate','download_only'):raise ValueError('invalid job mode')
    p=paths(root);key=str(p['root'].resolve())
    with _LOCK:
        if _JOBS.get(key) and _JOBS[key].is_alive():return {'started':False,'state':'RUNNING'}
        lockpath=p['root']/'v19_job.lock'
        if lockpath.exists():
            lock=read_json(lockpath)
            stale=time.time()-float(lock.get('at',0))>5400
            try:
                pid=int(lock.get('pid',0));pid_alive=False
                if pid>0:os.kill(pid,0);pid_alive=True
            except (OSError,ValueError):pid_alive=False
            if pid_alive and not stale and int(lock.get('pid',0))!=os.getpid():return {'started':False,'state':'ANOTHER_PROCESS'}
            lockpath.unlink(missing_ok=True)
        fd=os.open(lockpath,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd,'w') as f:
            import json
            json.dump({'pid':os.getpid(),'at':time.time()},f)
        runid=datetime.now(TZ).strftime('%Y%m%d_%H%M%S')+'_'+cfg.digest()
        directory=p['runroot']/runid
        state={'state':'RUNNING','mode':mode,'run_id':runid,'settings':asdict(cfg),'started_at':datetime.now(TZ).isoformat(),
               'pid':os.getpid(),'stage':'prepare','done':0,'total':1,'message':'Preparing data','updated_epoch':time.time()}
        atomic_json(p['state'],state)
        def progress(stage,done,total,message):
            state.update(stage=stage,done=done,total=total,message=message,updated_epoch=time.time())
            atomic_json(p['state'],state)
        def worker():
            try:
                if mode=='cache_validate':
                    prices,evidence=load_inputs(root,cfg);manifest=read_json(p['root']/'download_manifest.json')
                else:
                    prices,evidence,manifest=download_history(p['root'],cfg,progress)
                if prices.empty:raise SourceError('No price data obtained. See download_manifest.json; no fabricated results were generated.')
                if mode=='download_only':
                    state.update(state='DONE',message='History downloaded; choose cached validation next.',updated_epoch=time.time())
                else:
                    progress('features',0,1,'Building point-in-time price factors')
                    report,panel,trades=analyze(prices,evidence,cfg,progress,manifest)
                    save_analysis(directory,report,panel,trades)
                    if report.get('cutoffs'):
                        atomic_json(p['last'],{'directory':str(directory.resolve()),'run_id':runid,'status':report['status']})
                    state.update(state='DONE',message=report['status'],report_directory=str(directory.resolve()),updated_epoch=time.time())
            except Exception as e:
                state.update(state='FAILED',message=type(e).__name__+': '+str(e),updated_epoch=time.time())
            finally:
                atomic_json(p['state'],state)
                lockpath.unlink(missing_ok=True)
        t=threading.Thread(target=worker,daemon=True,name='AlphaRadarV19Research')
        _JOBS[key]=t;t.start()
    return {'started':True,'state':'RUNNING','run_id':runid}


def weights_for(report,horizon,asof):
    block=(report.get('weights') or {}).get(str(horizon)) or {}
    fit=block.get('fit') or {}
    if block.get('weights') and str(fit.get('cutoff','9999'))<=str(pd.Timestamp(asof).date()):
        return block['weights'],fit.get('status','RESEARCH_FIT')
    return prior(horizon,TECH),'STARTING_PRIOR_NO_FIT'


def analog_summary(panel,row,horizon,weights):
    """Past mature, spaced analogous dates. Historical frequencies, not forecasts."""
    asof=pd.Timestamp(row['date']);col=f'net_{horizon}'
    if col not in panel:return {'state':'NO_OUTCOMES','count':0}
    subset=panel[(panel.date<asof)&(panel[f'exit_date_{horizon}']<asof)&panel.eligible&panel[col].notna()].copy()
    keys=[k for k in weights if k in FACTORS and pd.notna(row.get(k))]
    if not keys or subset.empty:return {'state':'INSUFFICIENT','count':0}
    values=subset[keys].to_numpy(dtype=float);query=np.array([row[k] for k in keys]);w=np.array([weights[k] for k in keys])
    w=w/w.sum();mask=np.isfinite(values);coverage=mask@w
    distance=np.nansum((values-query)**2*w,axis=1)/np.maximum(coverage,.01)
    distance[coverage<.85]=np.inf
    subset['_distance']=distance
    subset=subset[np.isfinite(subset._distance)].sort_values('_distance').drop_duplicates('date')
    cal=pd.DatetimeIndex(panel.date.unique()).sort_values();chosen=[];locs=[]
    for _,item in subset.iterrows():
        loc=cal.get_loc(item.date)
        if all(abs(loc-q)>=horizon for q in locs):chosen.append(item);locs.append(loc)
        if len(chosen)>=40:break
    minimum=8 if horizon>=60 else 15
    if len(chosen)<minimum:return {'state':'INSUFFICIENT_SPACED_SAMPLES','count':len(chosen),'minimum':minimum}
    samples=pd.DataFrame(chosen);net=samples[col].to_numpy(dtype=float)
    return {'state':'HISTORICAL_ANALOG_NOT_CALIBRATED_FORECAST','count':len(samples),
            'mean_net':float(net.mean()),'median_net':float(np.median(net)),'positive_fraction':float((net>0).mean()),
            'q25_net':float(np.quantile(net,.25)),'worst_mae':float(samples[f'mae_{horizon}'].min()),
            'first':str(samples.date.min().date()),'last':str(samples.date.max().date()),'horizon':horizon}


def field_health(panel,report=None):
    if panel is None or panel.empty:return [{'field':'price history','present':0,'total':0,'state':'MISSING'}]
    current=panel[panel.date==panel.date.max()];out=[]
    for col in ('ticker','name','date','close','source','rsi','k','d','macd_hist','bias20','neckline')+FACTORS:
        present=int(current[col].notna().sum()) if col in current else 0
        out.append({'field':col,'present':present,'total':len(current),'coverage':present/len(current),
                    'state':'AVAILABLE' if present==len(current) else 'PARTIAL' if present else 'MISSING_NOT_ZERO'})
    kind=(report or {}).get('data_kind','NOT_RUN')
    out.append({'field':'real_market_validation','present':int(bool(report and report.get('cutoffs') and kind=='PROVIDER_HISTORY')),'total':1,
                'state':kind if kind in ('SYNTHETIC_TEST_ONLY','IMPORTED_UNVERIFIED') else (report or {}).get('status','NOT_RUN')})
    return out


def position_size(capital,entry,stop,risk_fraction=.005,max_weight=.2,cost_buffer=.007):
    if not all(np.isfinite(x) for x in (capital,entry,stop,risk_fraction,max_weight,cost_buffer)):
        return {'shares':0,'state':'INVALID_INPUT'}
    if capital<=0 or not 0<stop<entry or not 0<risk_fraction<=.05 or not 0<max_weight<=1:
        return {'shares':0,'state':'INVALID_INPUT'}
    per_share=entry-stop+entry*cost_buffer
    shares=int(min(capital*risk_fraction/per_share,capital*max_weight/(entry*(1+cost_buffer))))
    return {'shares':shares,'position_value':shares*entry,'planned_max_loss':shares*per_share,
            'risk_budget':capital*risk_fraction,'state':'ILLUSTRATION_GAP_RISK_NOT_CAPPED'}
