"""Collect per-fold output files after the documented analysis commands finish."""
from pathlib import Path
import argparse
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def collect(folder,prefix,output):
    files=sorted(folder.glob(f'{prefix}_seed*_fold*.csv'))
    if not files:
        files=sorted(p for p in folder.glob(f'{prefix}_seed*.csv') if '_fold' not in p.name)
    if not files:raise FileNotFoundError(f'No {prefix} seed/fold outputs in {folder}')
    z=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
    z.to_csv(output,index=False)
    print(f'{output.relative_to(ROOT)}: {len(z)} rows from {len(files)} files')
    return z

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--only',choices=['sentinel','transfer','partial','conformal'],nargs='+',default=['sentinel','transfer','partial','conformal'])
    args=p.parse_args();r=ROOT/'results'
    if 'sentinel' in args.only:
        folder=r/'sentinel_v3'
        base=collect(folder,'v3',folder/'v3_all.csv')
        extra=collect(folder,'extra',folder/'extra_all.csv')
        z=pd.concat([base,extra],ignore_index=True)
        z['strategy']=z.strategy.str.replace(r'random\d','random',regex=True)
        z.to_csv(folder/'v3_plus_extra_all.csv',index=False)
        metrics=[c for c in ['AUC','Brier','LogLoss','R2_logcum','R2_logRQ','AUC_AMR'] if c in z]
        z.groupby(['model','strategy','k'])[metrics].agg(['mean','std','count']).to_csv(folder/'summary_v3.csv')
    if 'transfer' in args.only:collect(r/'threshold_transfer','transfer',r/'threshold_transfer/transfer_all.csv')
    if 'conformal' in args.only:collect(r/'conformal','conformal',r/'conformal/conformal_all.csv')
    if 'partial' in args.only:
        folder=r/'partial_panels';files=sorted(folder.glob('partial_panels_fold*.csv'))
        if not files:
            single=folder/'partial_panels.csv';files=[single] if single.exists() else []
        if not files:raise FileNotFoundError('Run 07_partial_panels.py before collection.')
        pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(folder/'partial_panels_all.csv',index=False)

if __name__=='__main__':main()
