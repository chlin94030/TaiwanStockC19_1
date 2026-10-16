"""Ten seeded cutoff dates, causal nested walk-forward, auditable trade records.

Cutoff T -> entry T+1 open -> exit T+h close on a COMMON benchmark calendar.
Historical outcomes are admitted to training only after their exit < cutoff.
A holdout result never decides that same holdout's weights. No-fill slots stay
cash; selected missing exits invalidate portfolio returns rather than quietly
removing losing/delisted names. Overlapping dates are disclosed, not ten
independent experiments. Results describe this data universe, not global optima.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from v19_config import FACTORS, TECH, Settings, prior, VERSION
from v19_features import build_panel, rank_date, score_rows, current_candidates
from v19_data import atomic_json, atomic_csv


def net_return(entry,exit,cfg=Settings()):
    a=np.asarray(entry,dtype=float);b=np.asarray(exit,dtype=float)
    with np.errstate(divide='ignore',invalid='ignore'):
        value=b*(1-cfg.slippage)*(1-cfg.sell_fee-cfg.sell_tax)/(a*(1+cfg.slippage)*(1+cfg.buy_fee))-1
    return np.where((a>0)&(b>0)&np.isfinite(a)&np.isfinite(b),value,np.nan)


def calendar_for(panel,cfg):
    benchmark=panel[panel.ticker==cfg.benchmark].date.drop_duplicates().sort_values()
    if len(benchmark)>=cfg.warmup:
        counts=panel.groupby('date').ticker.nunique()
        dense=pd.DatetimeIndex(counts[counts>=max(2,int(panel.ticker.nunique()*.5))].index)
        return pd.DatetimeIndex(benchmark).union(dense).sort_values()
    # Fallback requires cross-section presence; never insert weekends via bdate_range.
    counts=panel.groupby('date').ticker.nunique()
    n=panel.ticker.nunique()
    return pd.DatetimeIndex(counts[counts>=max(2,int(n*.5))].index).sort_values()


def add_outcomes(panel,cfg=Settings()):
    cal=calendar_for(panel,cfg);pieces=[]
    for ticker,base in panel.groupby('ticker',sort=True):
        f=base.set_index('date').reindex(cal)
        c=f.calc_close.astype(float);o=f.calc_open.astype(float);lo=f.calc_low.astype(float);hi=f.calc_high.astype(float)
        raw=f.close.astype(float);vol=f.share_volume.astype(float);entry=o.shift(-1)
        rawopen=f.open.shift(-1)
        gap=rawopen/raw-1
        oneprice=(f.high.shift(-1)==f.low.shift(-1))&(gap.abs()>.085)
        nofill=rawopen.isna()|(vol.shift(-1)<=0)|(gap>cfg.max_entry_gap)|oneprice
        entry_state=np.where(rawopen.isna(),'NO_ENTRY_BAR',np.where(vol.shift(-1)<=0,'NO_VOLUME',
                    np.where(gap>cfg.max_entry_gap,'ABOVE_PREDECLARED_GAP_CAP',np.where(oneprice,'ONE_PRICE_SESSION','ASSUMED_OPEN_FILL'))))
        # Non-zero no-fill return is prohibited. It was an allocated cash slot.
        result=pd.DataFrame({'date':cal,'ticker':ticker,'entry_date':pd.Series(cal).shift(-1).to_numpy(),
                             'entry_open':rawopen.to_numpy(),'fill_state':entry_state})
        for h in cfg.horizons:
            end=c.shift(-h);exitdate=pd.Series(cal,index=cal).shift(-h)
            minimum=lo.shift(-1).iloc[::-1].rolling(h,min_periods=h).min().iloc[::-1]
            maximum=hi.shift(-1).iloc[::-1].rolling(h,min_periods=h).max().iloc[::-1]
            count=c.shift(-1).notna().astype(int).iloc[::-1].rolling(h,min_periods=h).sum().iloc[::-1]
            jumps=f.raw_discontinuity.eq(True).shift(-1,fill_value=False).astype(int)
            suspicious=jumps.iloc[::-1].rolling(h,min_periods=h).max().iloc[::-1].fillna(0).astype(bool)
            matured=exitdate.notna()
            exit_volume=vol.shift(-h)
            exit_gap=f.open.shift(-h)/raw.shift(-(h-1))-1
            exit_locked=(f.high.shift(-h)==f.low.shift(-h))&(exit_gap<-.085)
            ok=(count==h)&entry.notna()&end.notna()&(~suspicious)&(exit_volume>0)&(~exit_locked)
            cost_cfg=replace(cfg,sell_tax=cfg.benchmark_sell_tax) if ticker==cfg.benchmark else cfg
            net=np.asarray(net_return(entry,end,cost_cfg),dtype=float)
            net[~ok.to_numpy()]=np.nan
            net[(nofill&matured).to_numpy()]=0.
            net[~matured.to_numpy()]=np.nan
            mae=(minimum/entry-1).clip(upper=0).to_numpy();mfe=(maximum/entry-1).clip(lower=0).to_numpy()
            mae[~ok.to_numpy()]=np.nan;mfe[~ok.to_numpy()]=np.nan
            mae[(nofill&matured).to_numpy()]=0;mfe[(nofill&matured).to_numpy()]=0
            state=np.where(~matured,'NOT_MATURED',np.where(nofill,entry_state,
                     np.where(suspicious,'RAW_CORPORATE_ACTION_DISCONTINUITY',np.where(exit_locked,'EXIT_ONE_PRICE_DOWN_SESSION',
                     np.where(exit_volume<=0,'EXIT_NO_VOLUME',np.where(~ok,'MISSING_FORWARD_BAR','ASSUMED_OPEN_FILL'))))))
            # Position close-to-close max drawdown, separate from adverse excursion.
            mdd=np.full(len(cal),np.nan)
            if len(cal)>h:
                windows=np.lib.stride_tricks.sliding_window_view(c.to_numpy()[1:],h)
                starts=entry.to_numpy()[:len(windows)]
                combined=np.column_stack([starts,windows])
                peaks=np.maximum.accumulate(combined,axis=1)
                with np.errstate(divide='ignore',invalid='ignore'):
                    dd=combined/peaks-1
                mdd[:len(windows)]=np.min(dd,axis=1)
            mdd[~ok.to_numpy()]=np.nan;mdd[(nofill&matured).to_numpy()]=0.
            result[f'net_{h}']=net
            result[f'base_return_{h}']=(end/c-1).to_numpy()
            result[f'mae_{h}']=mae;result[f'mfe_{h}']=mfe;result[f'mdd_{h}']=mdd
            result[f'exit_date_{h}']=exitdate.to_numpy();result[f'exit_close_{h}']=f.close.shift(-h).to_numpy()
            result[f'outcome_state_{h}']=state
        pieces.append(result)
    outcomes=pd.concat(pieces,ignore_index=True)
    return panel.merge(outcomes,on=['ticker','date'],how='left',validate='one_to_one'),cal


def sample_cutoffs(panel,cal,cfg):
    counts=panel[panel.eligible].groupby('date').ticker.nunique()
    dates=pd.DatetimeIndex(counts[counts>=cfg.min_universe].index)
    latest_idx=len(cal)-1-max(cfg.horizons)
    if latest_idx<0:return []
    full_start=cfg.warmup+cfg.train_dates+max(cfg.horizons)+10
    min_start=min(full_start,max(cfg.warmup,latest_idx-cfg.dates*3))
    valid=[d for d in dates if cal.get_indexer([d])[0]>=min_start and d<=cal[latest_idx]]
    if len(valid)<cfg.dates:return []
    rng=np.random.default_rng(cfg.seed)
    # One draw per chronological stratum avoids ten coincident neighbouring dates.
    bins=np.array_split(np.arange(len(valid)),cfg.dates)
    return sorted(pd.Timestamp(valid[int(rng.choice(b))]) for b in bins)


def weight_candidates(horizon,features,cfg,seed):
    base=prior(horizon,features);w=np.array([base[k] for k in features])
    candidates=[base]
    # Single-group ablations test marginal contribution within TRAINING only.
    for i in range(len(features)):
        q=w.copy();q[i]=0
        if q.sum()>0:candidates.append(dict(zip(features,(q/q.sum()).tolist())))
        if len(candidates)>=cfg.trials:return candidates
    rng=np.random.default_rng(seed)
    while len(candidates)<cfg.trials:
        q=w*np.exp(rng.normal(0,.25,len(w)))
        q=np.maximum(q,0);q=q/q.sum()
        candidates.append(dict(zip(features,q.tolist())))
    return candidates


def group_performance(rows,weights,h,cfg):
    s,coverage=score_rows(rows,weights)
    q=rows.assign(_score=s).dropna(subset=['_score']).sort_values(['_score','ticker'],ascending=[False,True]).head(cfg.top_k)
    if len(q)<cfg.top_k:return np.nan,np.nan
    if q[f'net_{h}'].isna().any():return np.nan,np.nan
    return float(q[f'net_{h}'].mean()),float(q[f'mae_{h}'].mean())


def objective(groups,weights,h,cfg):
    values=[];down=[]
    for g in groups:
        r,m=group_performance(g,weights,h,cfg)
        if np.isfinite(r) and np.isfinite(m):values.append(r);down.append(m)
    if len(values)<max(5,int(len(groups)*.8)):return -np.inf,{'valid_dates':len(values),'total_dates':len(groups)}
    a=np.array(values);b=np.array(down)
    # Same-horizon candidate comparison, net of predefined costs. Penalties are
    # explicit preferences, not fitted constants reported as empirical truth.
    val=float(np.mean(a)+.25*np.mean(b)-.15*np.std(a))
    return val,{'objective':val,'mean_net':float(a.mean()),'mean_mae':float(b.mean()),'valid_dates':len(a),'total_dates':len(groups)}


def batch_objectives(groups,candidates,h,cfg):
    """Vectorized top-k replay. No per-trial pandas sort/groupby loops."""
    if not groups:return [(-np.inf,{'valid_dates':0,'total_dates':0}) for _ in candidates]
    keys=list(candidates[0]);n=len(groups);width=max(len(g) for g in groups)
    a=np.full((n,width,len(keys)),np.nan);y=np.full((n,width),np.nan);down=np.full_like(y,np.nan)
    penalty=np.zeros_like(y)
    for i,g in enumerate(groups):
        g=g.sort_values('ticker');m=len(g)
        a[i,:m,:]=g[keys].to_numpy(dtype=float)
        y[i,:m]=g[f'net_{h}'].to_numpy(dtype=float);down[i,:m]=g[f'mae_{h}'].to_numpy(dtype=float)
        penalty[i,:m]=np.clip(g.bias20.to_numpy(dtype=float)-12,0,15)*.7
    mask=np.isfinite(a);safe=np.nan_to_num(a)
    result=[]
    for weights in candidates:
        w=np.array([weights[k] for k in keys]);w=w/w.sum()
        coverage=mask@w
        raw=safe@w
        score=np.divide(raw,coverage,out=np.full_like(raw,-np.inf),where=coverage>=.75)-penalty
        score[~np.isfinite(score)]=-np.inf
        ids=np.argsort(-score,axis=1,kind='stable')[:,:cfg.top_k]
        selected=np.take_along_axis(y,ids,axis=1);adverse=np.take_along_axis(down,ids,axis=1)
        count=np.isfinite(np.take_along_axis(score,ids,axis=1)).sum(axis=1)
        ok=(count==cfg.top_k)&np.isfinite(selected).all(axis=1)&np.isfinite(adverse).all(axis=1)
        val_dates=int(ok.sum())
        if val_dates<max(5,int(n*.8)):
            result.append((-np.inf,{'valid_dates':val_dates,'total_dates':n}));continue
        means=selected[ok].mean(axis=1);mae=adverse[ok].mean(axis=1)
        obj=float(means.mean()+.25*mae.mean()-.15*means.std())
        result.append((obj,{'objective':obj,'mean_net':float(means.mean()),'mean_mae':float(mae.mean()),
                           'valid_dates':val_dates,'total_dates':n}))
    return result

def fit_before(panel,cutoff,h,cfg=Settings(),seed=1900):
    cutoff=pd.Timestamp(cutoff)
    # Strict maturity gate; outcomes ending on cutoff are not yet assumed known.
    available=panel[(panel.date<cutoff)&(panel[f'exit_date_{h}']<cutoff)&panel.eligible]
    grouped=[(d,g) for d,g in available.groupby('date',sort=True) if len(g)>=cfg.min_universe]
    info={'cutoff':str(cutoff.date()),'horizon':h,'status':'BASELINE_INSUFFICIENT_TRAINING','features':list(TECH),
          'holdout_used_for_selection':False,'train_dates':0,'validation_dates':0}
    if len(grouped)<cfg.train_dates+12:return prior(h,TECH),info,[]
    # Last chronological block is inner validation; earlier training outcomes
    # crossing its boundary are purged to prevent overlap leakage.
    val_n=max(12,min(40,len(grouped)//4));valid_pairs=grouped[-val_n:];valid_start=valid_pairs[0][0]
    train_pairs=[]
    for d,g in grouped[:-val_n]:
        g=g[g[f'exit_date_{h}']<valid_start]
        if len(g)>=cfg.min_universe:train_pairs.append((d,g))
    if len(train_pairs)<cfg.train_dates:return prior(h,TECH),info,[]
    train_pairs=train_pairs[-160:]
    training=pd.concat([g for _,g in train_pairs],ignore_index=True)
    coverage={k:float(training[k].notna().mean()) for k in FACTORS}
    features=[k for k in FACTORS if coverage[k]>=.85]
    if not all(k in features for k in TECH):return prior(h,TECH),dict(info,status='BASELINE_TECH_GAP'),[]
    candidates=weight_candidates(h,features,cfg,seed)
    traces=[]
    train_objectives=batch_objectives([g for _,g in train_pairs],candidates,h,cfg)
    for i,w in enumerate(candidates):
        obj,stats=train_objectives[i]
        traces.append({'candidate':i,'stage':'train','weights':w,**stats,'objective':obj})
    indices=sorted(range(len(candidates)),key=lambda i:traces[i]['objective'],reverse=True)[:min(6,len(candidates))]
    indices=sorted(set(indices+[0]))
    valid_scores={}
    valid_objectives=batch_objectives([g for _,g in valid_pairs],[candidates[i] for i in indices],h,cfg)
    for ix,i in enumerate(indices):
        obj,stats=valid_objectives[ix]
        valid_scores[i]=obj
        traces.append({'candidate':i,'stage':'inner_validation','weights':candidates[i],**stats,'objective':obj})
    best=max(valid_scores,key=valid_scores.get);base_obj=valid_scores[0]
    # A marginal/noisy validation gain is not sufficient to alter the baseline.
    if not np.isfinite(valid_scores[best]) or valid_scores[best]<base_obj+.002:best=0
    info.update(status='RESEARCH_FIT' if best else 'BASELINE_VALIDATION_NOT_BETTER',
                features=features,excluded={k:'training coverage below 85%' for k in FACTORS if k not in features},
                feature_coverage=coverage,train_dates=len(train_pairs),validation_dates=len(valid_pairs),
                train_first=str(train_pairs[0][0].date()),train_last=str(train_pairs[-1][0].date()),
                inner_validation_first=str(valid_start.date()),inner_validation_last=str(valid_pairs[-1][0].date()),
                candidate=best,validation_objective=valid_scores.get(best),baseline_objective=base_obj)
    return candidates[best],info,traces


def selected_result(panel,date,weights,h,cfg,strategy):
    q=rank_date(panel,date,weights,cfg).copy()
    record={'date':str(pd.Timestamp(date).date()),'horizon':h,'strategy':strategy,'selected':len(q),'slots':cfg.top_k}
    fields=['ticker','name','date','score','factor_coverage','entry_date','entry_open','fill_state',
            f'exit_date_{h}',f'exit_close_{h}',f'net_{h}',f'base_return_{h}',f'mae_{h}',f'mfe_{h}',f'mdd_{h}',f'outcome_state_{h}']
    trades=q.reindex(columns=fields).copy();trades['strategy']=strategy;trades['horizon']=h
    complete=len(q)==cfg.top_k and q[f'net_{h}'].notna().all()
    filled=int((q[f'outcome_state_{h}']=='ASSUMED_OPEN_FILL').sum()) if len(q) else 0
    record.update(complete=bool(complete),filled=filled,missing=int(q[f'net_{h}'].isna().sum()) if len(q) else cfg.top_k,
                  mean_net=float(q[f'net_{h}'].sum()/cfg.top_k) if complete else None,
                  win_fraction=float((q[f'net_{h}']>0).sum()/cfg.top_k) if complete else None,
                  mean_mae=float(q[f'mae_{h}'].mean()) if complete else None,
                  mean_position_mdd=float(q[f'mdd_{h}'].mean()) if complete else None)
    return record,trades


def analyze(prices,evidence=None,cfg=Settings(),progress=None,manifest=None):
    panel=build_panel(prices,evidence,cfg)
    panel,cal=add_outcomes(panel,cfg)
    dates=sample_cutoffs(panel,cal,cfg)
    sources=set(panel.source.astype(str))
    data_kind='SYNTHETIC_TEST_ONLY' if any('SYNTHETIC' in x for x in sources) else ('IMPORTED_UNVERIFIED' if any('unverified' in x for x in sources) else 'PROVIDER_HISTORY')
    common={'data_kind':data_kind,'version':VERSION,'settings':asdict(cfg),'data_start':str(panel.date.min().date()),
            'data_end':str(panel.date.max().date()),'symbols':int(panel.ticker.nunique()),'bars':len(panel),
            'calendar_sessions':len(cal),'manifest':manifest or {},'as_of':cfg.end(),
            'data_hash':hashlib.sha256(pd.util.hash_pandas_object(prices,index=False).values.tobytes()).hexdigest(),
            'is_buy_signal':False,'global_optimum_proven':False,
            'caveats':['Current/user universe is not verified historical membership; survivorship bias remains.',
                       'Adjusted-price coverage differs by provider. Raw price returns are not total returns.',
                       'Daily open fills and slippage are assumptions, not executed trades.',
                       'Ten dates can overlap, especially at 60 sessions; they are not independent trials.',
                       'No historical intraday path: daily low/open/close cannot prove a live reversal prediction.',
                       'No official limit-price series: limit-up hit probability is not reported.']}
    cutoffs=[str(x.date()) for x in dates]
    if not dates:
        return dict(common,status='INSUFFICIENT_HISTORY',cutoffs=[],reason='Not enough common sessions/universe for ten mature cutoffs',
                    results=[],weights={},fits=[],trials=[],latest=[]),panel,pd.DataFrame()
    records=[];trades=[];fits=[];trials=[];last_weights={}
    total=len(dates)*len(cfg.horizons);step=0
    for ix,date in enumerate(dates):
        for h in cfg.horizons:
            step+=1
            if progress:progress('validate',step,total,str(date.date())+f' / {h} sessions')
            w,info,trace=fit_before(panel,date,h,cfg,cfg.seed+ix*101+h)
            info['weights']=w;fits.append(info)
            for t in trace:trials.append(dict(t,cutoff=str(date.date()),horizon=h))
            baselines={'baseline':prior(h,list(w)),'adaptive':w}
            for label,weights in baselines.items():
                r,t=selected_result(panel,date,weights,h,cfg,label);records.append(r);trades.append(t)
            b=panel[(panel.date==date)&(panel.ticker==cfg.benchmark)]
            val=float(b[f'net_{h}'].iloc[0]) if len(b) and pd.notna(b[f'net_{h}'].iloc[0]) else None
            records.append({'date':str(date.date()),'horizon':h,'strategy':'benchmark_0050','mean_net':val,
                            'complete':val is not None,'selected':int(len(b))})
            # Equal-weight universe is a descriptive baseline, NOT hindsight top.
            universe=panel[(panel.date==date)&panel.eligible]
            ok=not universe.empty and universe[f'net_{h}'].notna().all()
            records.append({'date':str(date.date()),'horizon':h,'strategy':'equal_weight_universe',
                            'mean_net':float(universe[f'net_{h}'].mean()) if ok else None,
                            'complete':bool(ok),'selected':len(universe)})
    # Fit a deployable RESEARCH candidate at the latest available cutoff using
    # outcomes that have already matured. This does not overwrite test results.
    latest=[]
    for h in cfg.horizons:
        date=panel.date.max();w,info,trace=fit_before(panel,date,h,cfg,cfg.seed+999+h)
        last_weights[str(h)]={'weights':w,'fit':info,'not_approved_for_automatic_orders':True}
        q=current_candidates(panel,cfg,w,h)
        for r in q.head(10).to_dict('records'):latest.append(dict(r,horizon=h))
    complete=sum(r.get('complete',False) for r in records if r['strategy']=='adaptive')
    report=dict(common,status='RESEARCH_COMPLETE' if complete==total else 'RESEARCH_PARTIAL',
                cutoffs=cutoffs,complete_adaptive_cases=complete,expected_adaptive_cases=total,
                results=records,weights=last_weights,fits=fits,trials=trials,latest=latest,
                adjusted_bar_fraction=float(panel.adjusted_available.mean()),
                fit_status_counts=pd.Series([x['status'] for x in fits]).value_counts().to_dict())
    report['overlapping_horizons']={str(h):sum(cal.get_loc(dates[i])-cal.get_loc(dates[i-1])<h for i in range(1,len(dates))) for h in cfg.horizons}
    return report,panel,pd.concat(trades,ignore_index=True)


def report_markdown(report):
    out=['# Alpha Radar V19.0 - historical research report','',f"Status: **{report['status']}**",
         f"Data: {report.get('data_start')} to {report.get('data_end')}; {report.get('symbols')} symbols.",
         '','Entry: next common trading-session open. Exit: T+h close. Costs and gap cap are in settings.',
         'All results are research estimates, NOT actual fills or proof of optimal weights.','',
         '## Seeded cutoff dates','',', '.join(report.get('cutoffs') or []) or 'Insufficient dates','',
         '## Equal-weight portfolio net returns','',
         '| Cutoff | Sessions | Baseline | Adaptive | 0050 | Complete |','|---|---:|---:|---:|---:|---|']
    def pct(x):return '--' if x is None or not np.isfinite(x) else f'{x*100:+.2f}%'
    for d in report.get('cutoffs',[]):
        for h in report['settings']['horizons']:
            q={r['strategy']:r for r in report['results'] if r['date']==d and r['horizon']==h}
            out.append(f"| {d} | {h} | {pct(q.get('baseline',{}).get('mean_net'))} | {pct(q.get('adaptive',{}).get('mean_net'))} | {pct(q.get('benchmark_0050',{}).get('mean_net'))} | {q.get('adaptive',{}).get('complete',False)} |")
    out+=['','## Data classification','',report.get('data_kind','UNCLASSIFIED'),'','## Weight fit statuses','',json.dumps(report.get('fit_status_counts',{}),ensure_ascii=False),'','## Limitations']+['- '+s for s in report.get('caveats',[])]+['','## Configuration','', '```json',json.dumps(report['settings'],indent=2),'```']
    return '\n'.join(out)


def save_analysis(output,report,panel,trades):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'v19_results.json',report)
    atomic_json(output/'weights_candidate.json',{'version':VERSION,'as_of':report.get('data_end'),
                 'data_hash':report.get('data_hash'),'research_only':True,'weights':report.get('weights',{})})
    atomic_csv(output/'comparison.csv',pd.DataFrame(report.get('results',[])))
    atomic_csv(output/'selected_trades.csv',trades)
    atomic_csv(output/'factor_panel.csv.gz',panel)
    (output/'v19_validation.md').write_text(report_markdown(report),encoding='utf-8')
