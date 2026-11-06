"""Reconstruct measured-only summaries on saved Direct-LGBM folds, without fitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,roc_auc_score
from sklearn.model_selection import GroupKFold
import sentry_lib as S

ROOT=Path(__file__).resolve().parents[1]

def build(result_path):
    d=S.load();n=len(d['F']);apis=d['apis'];vo=d['v_obs']
    y,c,_=S.observed_endpoints(d,np.arange(n),d['pnec'])
    with np.errstate(divide='ignore',invalid='ignore'):
        ex=np.nan_to_num(vo/d['pnec'][None,:])>1
    r=pd.read_csv(result_path);rows=[]
    for seed in [0,1]:
        u=np.unique(d['groups']);perm=np.random.default_rng(seed).permutation(len(u))
        g=pd.Series(d['groups']).map(dict(zip(u,perm))).to_numpy()
        for fold,(tr,te) in enumerate(GroupKFold(5).split(np.zeros(n),groups=g)):
            freq=d['det'][tr].mean(0);erate=ex[tr].mean(0)
            targets={j for j in range(len(apis)) if d['det'][:,j].sum()>=5}
            pool={j for j in range(len(apis)) if freq[j]>=.05}&targets
            fr=[j for j in np.argsort(-freq) if j in pool];er=[j for j in np.argsort(-erate) if j in pool]
            union=[]
            for a,b in zip(er,fr):
                for j in [a,b]:
                    if j not in union:union.append(j)
            for strategy,k,expected in [('frequency',3,fr[:3]),('union_heur',6,union[:6])]:
                q=r[(r.model=='Direct-LGBM')&(r.seed==seed)&(r.fold==fold)&(r.strategy==strategy)&(r.k==k)]
                if len(q)!=1:raise ValueError(f'Expected one Direct result for {seed=}, {fold=}, {strategy=}; found {len(q)}')
                q=q.iloc[0];panel=str(q.panel).split('|');idx=[apis.index(a) for a in panel]
                if idx!=expected:raise ValueError(f'Saved panel differs from reconstructed training-side rule: {seed}, {fold}, {strategy}')
                measured=vo[te][:,idx].sum(1);binary=ex[te][:,idx].any(1).astype(int)
                row=dict(seed=seed,fold=fold,strategy=strategy,k=k,panel='|'.join(panel),n_test=len(te),n_positive=int(y[te].sum()),
                    measured_only_R2=r2_score(c[te],np.log10(1+measured)),measured_only_AUC=roc_auc_score(y[te],binary),
                    direct_R2=q.R2_logcum,direct_AUC=q.AUC,measured_concentration_share=measured.sum()/vo[te].sum(),
                    measured_exceedance_sensitivity=binary[y[te].astype(bool)].mean())
                row.update(delta_R2=row['direct_R2']-row['measured_only_R2'],delta_AUC=row['direct_AUC']-row['measured_only_AUC'])
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/sentinel_v3/v3_plus_extra_all.csv')
    p.add_argument('--output-dir',type=Path,default=ROOT/'tables')
    args=p.parse_args();z=build(args.results);args.output_dir.mkdir(parents=True,exist_ok=True)
    z.to_csv(args.output_dir/'Table_S17.csv',index=False)
    z.groupby(['strategy','k'])[[c for c in z.columns if c not in ['seed','fold','strategy','k','panel']]].agg(['mean','std']).to_csv(args.output_dir/'Table_S17_summary.csv')
    print(f'Wrote {len(z)} verified panel-fold rows to {args.output_dir}')

if __name__=='__main__':main()
