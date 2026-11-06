"""Fit complete panel footprints, including labels and colourbars, to compact cells."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.text import Text
from matplotlib.colors import to_rgba_array, to_rgb
from ml_publication_style import save_figure_with_subfigures
from scientific_palettes import get_palette, STYLE_COLORS

PALETTE = get_palette('au_nimno_soft_8')
INK = '#333333'

def register(fig, cell, letter, fn):
    before=set(fig.axes)
    ax=fig.add_subplot(cell); fn(ax)
    fig._panel_records=getattr(fig,'_panel_records',[])+[dict(letter=letter,axes=[a for a in fig.axes if a not in before],target=cell.get_position(fig).frozen())]
    return ax

def outer(fig, axes):
    renderer=fig.canvas.get_renderer()
    return Bbox.union([a.get_tightbbox(renderer) for a in axes if a.get_visible()]).transformed(fig.transFigure.inverted())

def refine(fig, letters=True):
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_fontweight('normal')
            t.set_fontsize(max(11.0,t.get_fontsize()))
    for rec in fig._panel_records:
        for a in rec['axes']: a.set_axes_locator(None)
    report=[]
    for rec in fig._panel_records:
        target=rec['target']; axes=rec['axes']
        for _ in range(28):
            fig.canvas.draw(); box=outer(fig,axes)
            residual=(np.array(box.extents)-target.extents)*np.tile(fig.get_size_inches(),2)*72
            if np.max(np.abs(residual))<0.35: break
            sx=target.width/box.width;sy=target.height/box.height
            for a in axes:
                p=a.get_position(original=True)
                a.set_position([target.x0+(p.x0-box.x0)*sx,target.y0+(p.y0-box.y0)*sy,p.width*sx,p.height*sy],which='both')
        fig.canvas.draw();box=outer(fig,axes)
        residual=(np.array(box.extents)-target.extents)*np.tile(fig.get_size_inches(),2)*72
        report.append(dict(panel=rec['letter'],outer=list(box.extents),target=list(target.extents),max_residual_pt=float(np.max(np.abs(residual)))))
    if letters:
        w,h=fig.get_size_inches()*72
        for rec in report:
            x0,y0,x1,y1=rec['outer']
            fig.text(x0-6/w,y1,rec['panel'],ha='right',va='top',fontsize=16,fontweight='normal',color=INK)
    return report

def artist_audit(fig):
    approved=[to_rgb(p) for p in PALETTE];bad=[];n=0
    for a in fig.findobj():
        if isinstance(a,Text): continue
        if hasattr(a,'get_array') and a.get_array() is not None: continue
        for getter in ['get_color','get_facecolor','get_edgecolor','get_markerfacecolor','get_markeredgecolor']:
            if not hasattr(a,getter): continue
            try: colors=to_rgba_array(getattr(a,getter)())
            except Exception: continue
            for rgba in colors:
                if rgba[3]==0:continue
                rgb=rgba[:3];n+=1
                neutral=np.max(rgb)-np.min(rgb)<1e-6 or np.allclose(rgb,to_rgb(STYLE_COLORS['ink']),atol=1e-6)
                if not neutral and not any(np.allclose(rgb,c,atol=1e-6) for c in approved):bad.append(dict(artist=type(a).__name__,getter=getter,rgb=rgb.tolist()))
    return dict(palette='au_nimno_soft_8',checked=n,unexpected_colors=bad,nonregular_text=[t.get_text() for t in fig.findobj(Text) if t.get_text() and t.get_fontweight() not in ['normal',400]])

def save(fig,folder,stem,panels):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=refine(fig);audit=artist_audit(fig)
    save_figure_with_subfigures(fig,folder,stem,export_panels=False)
    sub=folder/'subfigures';sub.mkdir(exist_ok=True)
    for letter,fn,size in panels:
        f=plt.figure(figsize=size);g=f.add_gridspec(1,1,left=.07,right=.98,bottom=.07,top=.98)
        register(f,g[0,0],letter,fn);refine(f,letters=False)
        save_figure_with_subfigures(f,sub,stem+'_'+letter,export_panels=False)
    (sub/'README.md').write_text('# Subfigures\n\nEach panel is independently rendered on a standalone canvas from the same panel builder as the combined figure; panels are not cropped.\n',encoding='utf8')
    qa=folder.parents[1]/'results'/'figure_layout';qa.mkdir(parents=True,exist_ok=True)
    (qa/(stem+'.json')).write_text(json.dumps(dict(alignment=report,artist_audit=audit),indent=2),encoding='utf8')
    print(stem, 'max alignment residual',round(max(r['max_residual_pt'] for r in report),3),'pt; unexpected colors',len(audit['unexpected_colors']),flush=True)
