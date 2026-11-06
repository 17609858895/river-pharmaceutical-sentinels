"""Render the ten main figures from existing analyses; never refit models.

Usage: python scripts/figures_publication.py [--only 1 3 ...]
"""
from pathlib import Path
import sys,json,argparse
if __name__ == '__main__' and ('--help' in sys.argv or '-h' in sys.argv):
    print(__doc__)
    print('--only accepts figure numbers 1 through 10. Processed data and result CSVs must exist before rendering.')
    raise SystemExit(0)
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'figlib')]
from ml_publication_style import configure_publication_style,style_axis,numeric_ticks,export_subfigures,save_figure_with_subfigures
from scientific_palettes import get_palette
from project_figure_layout import register,save
import sentry_lib as S
import fig01_landscape as L
import fig03_fig07 as N
import fig04_panel_design as P
import fig05_fig06 as A

configure_publication_style();pal=get_palette('au_nimno_soft_8');INK='#333333';ROOT=S.ROOT
plt.rcParams.update({'axes.labelweight':'normal','font.weight':'normal','axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12})
R=ROOT/'results'; FIG=ROOT/'figures'

def make(stem,panels,cells,size):
    f=plt.figure(figsize=size)
    nr=max(x[0]+x[2] for x in cells);nc=max(x[1]+x[3] for x in cells)
    g=f.add_gridspec(nr,nc,left=.035,right=.99,bottom=.035,top=.985,wspace=.10,hspace=.12)
    for (letter,fn,_),(row,col,rs,cs) in zip(panels,cells):register(f,g[row:row+rs,col:col+cs],letter,fn)
    save(f,FIG/stem,stem,panels)

def fig1():
    make('Fig01_landscape',L.PANELS,[(0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,1,1)],(12.4,9.3))

def fig2():
    import runpy
    runpy.run_path(str(HERE/'fig02_schematic.py'),run_name='__main__')

def fig3():
    ps=[('a',lambda a:N.bench_panel(a,'log_cum','R2_mean','Concentration R$^2$',-.4),(6.6,7.2)),('b',lambda a:N.bench_panel(a,'risk_any','AUC_mean','Exceedance ROC AUC',.5),(6.6,7.2)),('c',known_panel_gain,(8.4,3.5))]
    f=plt.figure(figsize=(12.6,10.1));g=f.add_gridspec(3,2,left=.035,right=.99,bottom=.035,top=.94,wspace=.10,hspace=.12,height_ratios=[1,1,.68])
    a=register(f,g[:2,0],'a',lambda ax:N.bench_panel(ax,'log_cum','R2_mean','Concentration R$^2$',-.4,legend=False))
    register(f,g[:2,1],'b',lambda ax:N.bench_panel(ax,'risk_any','AUC_mean','Exceedance ROC AUC',.5,legend=False))
    register(f,g[2,:],'c',known_panel_gain)
    handles,labels=a.get_legend_handles_labels()
    f.legend(handles,labels,loc='upper center',bbox_to_anchor=(.51,.995),ncol=3,fontsize=11,columnspacing=1.5,handletextpad=.4)
    save(f,FIG/'Fig03_information_bottleneck','Fig03_information_bottleneck',ps)

def known_panel_gain(ax):
    z=pd.read_csv(R/'sentinel_v3'/'v3_plus_extra_all.csv');z=z[z.model=='Direct-LGBM']
    keys=[('none',0),('frequency',3),('union_heur',6)]
    for metric,col,mk,lab in [('AUC',pal[1],'o','Exceedance ROC AUC'),('R2_logcum',pal[0],'s','Concentration R$^2$')]:
        v=[z[(z.strategy==st)&(z.k==k)][metric] for st,k in keys]
        ax.errorbar(range(3),[a.mean() for a in v],yerr=[a.std() for a in v],color=col,marker=mk,mec=INK,ms=8,lw=2,capsize=3,label=lab)
    ax.set_xticks(range(3));ax.set_xticklabels(['Covariates only','Three detection-ranked drugs','Six alternating sentinels'])
    style_axis(ax,'Measured information','Held-out score',label_size=14,tick_size=12);numeric_ticks(ax,x=False,n=4)
    ax.legend(fontsize=11,loc='lower right')

def fig4():
    ps=[('a',lambda a:P.curve(a,'AUC','Exceedance ROC AUC',legend=True),(6,4.5)),('b',lambda a:P.curve(a,'R2_logcum','Concentration R$^2$'),(6,4.5)),('c',P.pareto,(8.4,4.5))]
    make('Fig04_panel_design',ps,[(0,0,1,1),(0,1,1,1),(1,0,1,2)],(11.7,8.4))

def component(ax,metric,label):
    df=pd.read_csv(R/'sentinel_v3'/'v3_plus_extra_all.csv')
    idx=['seed','fold','strategy','k','panel'];a=df[df.model=='SENTRY'][idx+[metric]]
    b=df[df.model=='SENTRY-indep'][idx+[metric]];c=df[df.model=='Substitution'][idx+[metric]]
    z=a.merge(b,on=idx,suffixes=('_S','_I')).merge(c,on=idx)
    vals=[z[metric+'_S'],z[metric+'_I'],z[metric]]
    names=['Censored\n+ dependence','Censored\n+ independence','LOD/2\n+ dependence']
    for j,(v,col) in enumerate(zip(vals,[pal[1],pal[0],pal[4]])):
        # Physical points show all matched outer-fold/panel results; no inferential CI.
        jitter=np.linspace(-.12,.12,len(v))
        ax.scatter(np.full(len(v),j)+jitter,v,s=12,color=col,edgecolors=INK,linewidths=.2,alpha=.45,zorder=2)
        ax.errorbar(j,v.mean(),yerr=v.std(),fmt='D',ms=9,color=col,mec=INK,mew=.8,elinewidth=1.8,capsize=5,zorder=3)
    ax.axhline(0,color=INK,lw=.8,ls=':');ax.set_xticks(range(3));ax.set_xticklabels(names)
    style_axis(ax,None,label,tick_size=11.5,label_size=14);numeric_ticks(ax,x=False,n=5)
    ax.text(.98,.03,f'n = {len(z)} matched evaluations',transform=ax.transAxes,ha='right',fontsize=11)

def fig5():
    ps=[('a',lambda a:component(a,'R2_logcum','Concentration R$^2$'),(6,4.5)),('b',lambda a:component(a,'R2_logRQ','ΣRQ R$^2$'),(6,4.5))]
    make('Fig05_censoring_aware_inference',ps,[(0,0,1,1),(0,1,1,1)],(11.7,4.6))

def fig6():
    ps=[('a',A.p5b,(6.3,4.8)),('b',A.p5c,(5.8,4.8)),('c',amr_transfer,(8.5,4.6))]
    make('Fig06_threshold_transfer',ps,[(0,0,1,1),(0,1,1,1),(0,2,1,1)],(12.7,4.6))

def amr_transfer(ax):
    z=A.tt[A.tt.endpoint=='PNEC-AMR'].copy()
    z['k']=z.panel.map({'none':0,'union_k3':3,'union_k6':6})
    for model,col,mk in [('SENTRY',pal[1],'o'),('Direct-transfer',pal[0],'s'),('Direct-retrained',pal[2],'^')]:
        g=z[z.model==model].groupby('k').AUC.agg(['mean','std'])
        ax.errorbar(g.index,g['mean'],yerr=g['std'],color=col,marker=mk,mec=INK,ms=8,lw=2,capsize=3,label=model)
    style_axis(ax,'Measured sentinels (k)','AMR ROC AUC',label_size=14)
    ax.set_xticks([0,3,6]);numeric_ticks(ax,x=False,n=5);ax.legend(fontsize=11,loc='lower right')

def fig7():
    ps=[('a',A.p6a,(6,5.2)),('b',A.p6b,(6.5,5.2))]
    make('Fig07_hidden_risk',ps,[(0,0,1,1),(0,1,1,1)],(12.0,5.1))

CF=pd.read_csv(R/'conformal'/'conformal_all.csv')
def cf_curve(ax,metric,label):
    c=CF[(CF.strategy=='union_heur')&(CF.k==6)]
    for model,col,mk in [('Direct-LGBM',pal[0],'s'),('SENTRY',pal[1],'o')]:
        g=c[c.model==model].groupby('alpha')[metric].agg(['mean','std'])
        ax.errorbar(g.index,g['mean'],yerr=g['std'],color=col,marker=mk,mec=INK,ms=8,capsize=3,lw=2,label=model)
    if metric=='miss':ax.plot([0,.22],[0,.22],ls=':',color=INK,lw=1)
    ax.set_xlim(.025,.225);ax.set_xticks([.05,.1,.2]);ax.set_ylim(0,.45 if metric=='miss' else 1.10)
    style_axis(ax,'Nominal miss rate α',label,label_size=14);numeric_ticks(ax,x=False,n=5)
    if metric == 'miss':
        ax.set_ylim(-.025, .30); ax.set_yticks([0, .1, .2, .3])
    ax.legend(fontsize=11,loc='upper left' if metric=='miss' else 'lower right')

def cf_distribution(ax):
    c=CF[CF.alpha==.1]
    pairs=[('none',0,'No sentinels'),('union_heur',6,'Six sentinels')]
    for q,(strategy,k,name) in enumerate(pairs):
        for j,(model,col,mk) in enumerate([('Direct-LGBM',pal[0],'s'),('SENTRY',pal[1],'o')]):
            v=c[(c.strategy==strategy)&(c.k==k)&(c.model==model)].miss.to_numpy()
            x=q+(j-.5)*.23
            ax.scatter(x+np.linspace(-.035,.035,len(v)),v,s=45,c=col,edgecolors=INK,linewidths=.5,marker=mk,label=model if q==0 else None)
            ax.plot([x-.07,x+.07],[v.mean()]*2,color=col,lw=3)
    ax.axhline(.1,color=INK,lw=1,ls=':');ax.set_xticks([0,1]);ax.set_xticklabels([p[2] for p in pairs])
    style_axis(ax,None,'Held-out miss rate (α = 0.10)',label_size=14);numeric_ticks(ax,x=False,n=5)
    ax.legend(fontsize=11,loc='upper right')

def fig8():
    ps=[('a',lambda a:cf_curve(a,'miss','Held-out miss rate'),(6,4.5)),('b',lambda a:cf_curve(a,'cleared_safe','Safe sites cleared'),(6,4.5)),('c',cf_distribution,(7.5,4.5))]
    make('Fig08_screening',ps,[(0,0,1,1),(0,1,1,1),(0,2,1,1)],(12.7,4.6))

def fig9():
    make('Fig09_information_network',[('a',N.net_panel,(7.2,6.4)),('b',N.hub_panel,(5.5,6.2))],[(0,0,1,1),(0,1,1,1)],(12.0,6.1))

LABELS={'log_regional_pop':'Regional population','log_dist_centroid_km':'Distance from campaign centre','log_EC_uScm':'Electrical conductivity','sanitation_pct':'Sanitation coverage','rel_position':'Within-campaign river position','season_cos':'Season (cosine)','geo_x':'Geographic coordinate','year':'Sampling year','kw_wastewater':'Wastewater field note','death_rate':'Mortality rate','season_sin':'Season (sine)','log_GNI':'GNI per capita'}
IMP=pd.read_csv(R/'interpret'/'covariate_importance_k0.csv',index_col=0).iloc[:,0]
DEP=pd.read_csv(R/'interpret'/'covariate_dependence_k0.csv')
def cov_importance(ax):
    z=IMP.sort_values(ascending=False).head(10).iloc[::-1]
    ax.barh(range(len(z)),z.values,color=pal[0],edgecolor=INK,lw=.6)
    ax.set_yticks(range(len(z)));ax.set_yticklabels([LABELS.get(x,x) for x in z.index])
    style_axis(ax,'Mean |SHAP value|',None,tick_size=11,label_size=14);numeric_ticks(ax,y=False,n=4)

def cov_dependence(ax,feature,xlabel):
    z=DEP[['x_'+feature,'s_'+feature]].dropna();x=z.iloc[:,0];y=z.iloc[:,1]
    # Quantile summaries of existing SHAP values; no response/causal curve is fitted.
    bins=pd.qcut(x,14,duplicates='drop');q=z.groupby(bins,observed=True).agg(['median',lambda v:np.quantile(v,.25),lambda v:np.quantile(v,.75)])
    xm=q.iloc[:,0].to_numpy();ym=q.iloc[:,3].to_numpy();lo=q.iloc[:,4].to_numpy();hi=q.iloc[:,5].to_numpy()
    ax.fill_between(xm,lo,hi,color=pal[3],alpha=.7,lw=0)
    ax.plot(xm,ym,marker='o',color=pal[0],mec=INK,ms=5,lw=1.6)
    ax.axhline(0,color=INK,lw=.8,ls=':');style_axis(ax,xlabel,'SHAP value (ln concentration)',label_size=13,tick_size=11.5);numeric_ticks(ax,n=4)

def fig10():
    ps=[('a',cov_importance,(6.7,4.8)),('b',lambda a:cov_dependence(a,'log_regional_pop','log$_{10}$(regional population)'),(5.6,4.8)),('c',lambda a:cov_dependence(a,'log_EC_uScm','log$_{10}$(1 + EC [μS cm$^{-1}$])'),(5.8,4.8)),('d',N.dml_panel,(6.4,4.8))]
    make('Fig10_context',ps,[(0,0,1,1),(0,1,1,1),(1,0,1,1),(1,1,1,1)],(12,8.7))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='*',type=int);args=p.parse_args()
    for n,fn in [(1,fig1),(2,fig2),(3,fig3),(4,fig4),(5,fig5),(6,fig6),(7,fig7),(8,fig8),(9,fig9),(10,fig10)]:
        if args.only is None or n in args.only:fn()
