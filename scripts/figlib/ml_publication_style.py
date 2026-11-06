"""Small, self-contained plotting helpers for the released study figures."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def configure_publication_style():
    plt.rcParams.update({'font.family':'Arial','font.sans-serif':['Arial','DejaVu Sans'],
        'axes.unicode_minus':False,'figure.dpi':160,'savefig.dpi':600,
        'axes.labelsize':15.5,'axes.labelweight':'normal','font.weight':'normal',
        'xtick.labelsize':12.5,'ytick.labelsize':12.5,'legend.fontsize':11.5,
        'axes.linewidth':1.7,'xtick.major.size':7.2,'ytick.major.size':7.2,
        'xtick.major.width':1.55,'ytick.major.width':1.55,
        'xtick.minor.size':4.2,'ytick.minor.size':4.2,
        'xtick.minor.width':1.15,'ytick.minor.width':1.15,
        'xtick.direction':'out','ytick.direction':'out','legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42})

def style_axis(ax,xlabel=None,ylabel=None,tick_size=12.5,label_size=15.5):
    ax.grid(False)
    for key in ['top','right']: ax.spines[key].set_visible(False)
    for key in ['bottom','left']:
        ax.spines[key].set_visible(True);ax.spines[key].set_color('#30343B');ax.spines[key].set_linewidth(1.7)
    ax.xaxis.set_ticks_position('bottom');ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x',which='major',bottom=True,top=False,colors='#30343B',width=1.55,length=7.2,direction='out',labelsize=tick_size)
    ax.tick_params(axis='y',which='major',left=True,right=False,colors='#30343B',width=1.55,length=7.2,direction='out',labelsize=tick_size)
    ax.tick_params(axis='both',which='minor',colors='#30343B',width=1.15,length=4.2,direction='out')
    for axis in [ax.xaxis,ax.yaxis]:
        for tick in axis.get_major_ticks():
            tick.tick1line.set_visible(True);tick.tick1line.set_markersize(7.2);tick.tick1line.set_markeredgewidth(1.55)
    for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontweight('normal');t.set_fontsize(tick_size)
    ax.xaxis.label.set_fontweight('normal');ax.yaxis.label.set_fontweight('normal')
    ax.xaxis.label.set_size(label_size);ax.yaxis.label.set_size(label_size)
    if xlabel is not None:ax.set_xlabel(xlabel,fontweight='normal',fontsize=label_size,labelpad=7)
    if ylabel is not None:ax.set_ylabel(ylabel,fontweight='normal',fontsize=label_size,labelpad=7)

def numeric_ticks(ax,x=True,y=True,n=5):
    if x:ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
    if y:ax.yaxis.set_major_locator(MaxNLocator(nbins=n))

def export_subfigures(*args,**kwargs):
    raise RuntimeError('Use project_figure_layout.save: panels are independently rendered by their builders.')

def save_figure_with_subfigures(fig,folder,stem,dpi=600,close=True,export_panels=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f'{stem}.png',dpi=dpi,bbox_inches='tight',facecolor='white',metadata={'Software':'Matplotlib'})
    fig.savefig(folder/f'{stem}.pdf',bbox_inches='tight',facecolor='white',metadata={'Creator':'Matplotlib','Producer':'Matplotlib'})
    if export_panels:export_subfigures()
    if close:plt.close(fig)
