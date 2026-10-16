"""CLI: python run_v19.py --years 3 --stocks 100 --dates 10 --horizons 5,20,60"""
from __future__ import annotations
import argparse
from pathlib import Path
from v19_config import Settings
from v19_data import download_history,read_json,atomic_json
from v19_service import load_inputs
from v19_backtest import analyze,save_analysis

def main(argv=None):
    p=argparse.ArgumentParser(description='Alpha Radar V19.0 public-history walk-forward research')
    p.add_argument('--years',type=int,default=3);p.add_argument('--stocks',type=int,default=100)
    p.add_argument('--dates',type=int,default=10);p.add_argument('--horizons',default='5,20,60')
    p.add_argument('--trials',type=int,default=24);p.add_argument('--seed',type=int,default=1900)
    p.add_argument('--as-of',default='');p.add_argument('--data',default='data/v19')
    p.add_argument('--output',default='data/v19/cli_report');p.add_argument('--cache-only',action='store_true')
    p.add_argument('--optional-data',action='store_true');p.add_argument('--tickers',default='')
    a=p.parse_args(argv)
    cfg=Settings(years=a.years,stocks=a.stocks,dates=a.dates,horizons=tuple(int(x) for x in a.horizons.split(',')),
                 trials=a.trials,seed=a.seed,as_of=a.as_of,optional_data=a.optional_data)
    def progress(stage,i,total,msg):print(f'[{stage}] {i}/{total}: {msg}',flush=True)
    if a.cache_only:
        prices,evidence=load_inputs(a.data,cfg);manifest=read_json(Path(a.data)/'download_manifest.json')
    else:
        tickers=[s.strip() for s in a.tickers.split(',') if s.strip()] or None
        prices,evidence,manifest=download_history(a.data,cfg,progress,tickers=tickers)
    if prices.empty:
        atomic_json(Path(a.output)/'run_status.json',{'status':'NO_PRICE_DATA','source_manifest':manifest})
        print('No prices received. Inspect source manifest; caches and earlier reports have been preserved.')
        return 2
    report,panel,trades=analyze(prices,evidence,cfg,progress,manifest)
    save_analysis(a.output,report,panel,trades)
    print(report['status']+' | '+str(Path(a.output).resolve()))
    return 0 if report.get('cutoffs') else 3

if __name__=='__main__':raise SystemExit(main())
