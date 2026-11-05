# Figure 2 (v3): development disagreement vs. probe AUROC for all 31 settings.
# Data transcribed from tab:dev_full, tab:mmlu_small_medium, Appendix G (informed/stronger helper, SQuAD).
# Labels: each names a model pair and benchmark (or, in panel b, what changed) and is tied by thin lines to its
# own settings (blue Text, orange C2C); label positions are fixed coordinates chosen so that no label, line,
# or point overlaps and every line is 7 to 30 pt long.
# Run from any directory: python3 code/figures/fig2_boundary.py [output.pdf]
# (default output: figures/figure2_boundary.pdf; identical to the paper's Figure 2)
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family":"STIXGeneral","mathtext.fontset":"stix","font.size":7.5,
                     "axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
                     "xtick.labelsize":7,"ytick.labelsize":7})
TEXT="#2a78d6"; C2C="#eb6834"; BAND="#E3F4EA"; INK="#2b2b2b"; MUTED="#6b6b6b"
# (group, path, disagreement %, AUROC, deployed)
qwen=[
 ("small","T",100*339/742,0.641,0),("small","C",100*350/742,0.700,0),
 ("small","T",100*145/299,0.634,0),("small","C",100*160/299,0.641,0),
 ("small","T",46.76,0.644,0),("small","C",60.24,0.626,0),
 ("medium","T",100*174/742,0.787,0),("medium","C",100*131/742,0.840,1),
 ("medium","T",100*52/299,0.742,0),("medium","C",100*48/299,0.910,1),
 ("medium","T",42.71,0.717,0),("medium","C",53.39,0.765,0),
 ("large","T",100*72/742,0.873,1),("large","C",100*60/742,0.883,1),
 ("large","T",100*15/299,0.952,1),("large","C",100*12/299,0.909,1),
 ("large","T",100*561/2641,0.821,1),("large","C",100*756/2641,0.846,1)]
ext=[
 ("OLMo","T",100*172/742,0.790,0),("OLMo","T",100*67/299,0.749,0),("OLMo","T",100*1112/2641,0.633,0),
 ("L1B","T",100*191/742,0.693,0),("L1B","C",100*427/742,0.618,0),("L1B","T",100*88/299,0.678,0),("L1B","C",100*174/299,0.615,0),
 ("L8B","T",100*86/742,0.874,1),("L8B","T",100*37/299,0.880,1),
 ("fact","T",10.2,0.889,1),("strong","T",21.2,0.792,0),("strong","T",13.4,0.800,0),
 ("squad","T",15.4,0.811,1)]
assert len(qwen)+len(ext)==31
dep=[p for p in qwen+ext if p[4]]; fb=[p for p in qwen+ext if not p[4]]
assert len(dep)==12 and min(p[3] for p in dep)>=0.81 and max(p[3] for p in fb)<=0.80
fig,axes=plt.subplots(1,2,figsize=(5.5,2.3),sharey=True,gridspec_kw=dict(width_ratios=[1.3,1]))
for ax,data,title in [(axes[0],qwen,"(a) Qwen pairs (18 main settings)"),(axes[1],ext,"(b) Other receivers, helpers, and tasks (13)")]:
    ax.set_xscale("log"); ax.set_xlim(3.3,100); ax.set_ylim(0.57,0.985)
    ax.axhspan(0.805,0.985,color=BAND,zorder=0,lw=0)
    ax.axhline(0.805,color="#7cc49a",lw=0.6,ls=(0,(3,2)),zorder=1)
    ax.set_xticks([5,10,20,50]); ax.set_xticklabels(["5","10","20","50"])
    ax.minorticks_off()
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.grid(axis="y",color="#e6e6e6",lw=0.4,zorder=0)
    for g,pth,x,y,d in data:
        col=TEXT if pth=="T" else C2C
        ax.scatter([x],[y],s=26,marker="o",facecolor=col if d else "white",edgecolor=col,lw=1.1,zorder=3)
    ax.set_title(title,fontsize=7.5,loc="left",pad=3)
    ax.set_xlabel("Disagreement with deployed reference (%)",fontsize=7.5)
axes[0].set_ylabel("AUROC of the probe",fontsize=7.5)
handles=[Line2D([],[],marker="o",ls="",markerfacecolor=TEXT,markeredgecolor=TEXT,ms=4.5,label="Text"),
         Line2D([],[],marker="o",ls="",markerfacecolor=C2C,markeredgecolor=C2C,ms=4.5,label="C2C"),
         Line2D([],[],marker="o",ls="",markerfacecolor="#555",markeredgecolor="#555",ms=4.5,label="certified"),
         Line2D([],[],marker="o",ls="",markerfacecolor="white",markeredgecolor="#555",ms=4.5,label="falls back"),
         Patch(facecolor=BAND,edgecolor="#7cc49a",lw=0.5,ls="--",label="AUROC $\\geq$ 0.81: all 12 certified settings")]
fig.legend(handles=handles,loc="upper center",ncol=5,frameon=False,fontsize=7,bbox_to_anchor=(0.5,1.02),handlelength=1.0,handleheight=0.9,handletextpad=0.5,columnspacing=1.8)
fig.tight_layout(rect=(0,0,1,0.93),w_pad=1.0)   # layout is fixed before the labels are added
a,b=axes
LEAD="#8a8a8a"
P=lambda n,d:100*n/d
LABELS=[]
def lab(ax,txt,targets,xyt):
    """Label centered at xyt (drawn once), tied by a thin line to each target; lines run behind the markers."""
    LABELS.append((ax,ax.text(xyt[0],xyt[1],txt,fontsize=6.5,ha="center",va="center",color=INK,zorder=5),targets))
def draw_leaders():
    """Each line starts 3.2 pt outside the label's text box and stops 3.3 pt short of the marker center."""
    fig.canvas.draw(); rend=fig.canvas.get_renderer(); pt=fig.dpi/72
    for ax,tx,targets in LABELS:
        bb=tx.get_window_extent(rend).padded(2*pt); c=np.array([(bb.x0+bb.x1)/2,(bb.y0+bb.y1)/2])
        for t in targets:
            p=ax.transData.transform(t); d=p-c; n=np.hypot(*d); u=d/n
            e=min((bb.width/2)/abs(u[0]) if u[0] else np.inf,(bb.height/2)/abs(u[1]) if u[1] else np.inf)
            s=c+u*(e+1.2*pt); f=p-u*3.3*pt
            (x0,y0),(x1,y1)=ax.transData.inverted().transform([s,f])
            ax.plot([x0,x1],[y0,y1],color=LEAD,lw=0.45,zorder=2,solid_capstyle="butt")
# panel (a)
lab(a,"large/ARC",[(P(15,299),0.952),(P(12,299),0.909)],(4.472,0.838))
lab(a,"large/OBQA",[(P(72,742),0.873),(P(60,742),0.883)],(11.74,0.963))
lab(a,"large/MMLU-Pro",[(P(756,2641),0.846),(P(561,2641),0.821)],(60,0.832))
lab(a,"medium/ARC",[(P(48,299),0.910)],(10.0,0.833))
lab(a,"medium/ARC",[(P(52,299),0.742)],(15.2,0.638))
lab(a,"medium/OBQA C2C:\ncertified",[(P(131,742),0.840)],(36,0.905))
lab(a,"medium/OBQA Text:\nmore accurate, not certified",[(P(174,742),0.787)],(8.1,0.749))
lab(a,"medium/\nMMLU-Pro",[(53.39,0.765),(42.71,0.717)],(26,0.725))
tx,ty=P(174,742),0.787; cx,cy=P(131,742),0.840
a.annotate("",xy=(cx,cy),xytext=(tx,ty),
           arrowprops=dict(arrowstyle="-|>",lw=0.8,color=INK,mutation_scale=6,shrinkA=4.5,shrinkB=4.5,connectionstyle="arc3,rad=-0.35"),zorder=4)
# small pair: a brace groups its six settings; one line ties the brace to the label
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch
def brace(ax,x0,y0,y1,xm,xt,q):
    ym=(y0+y1)/2
    v=[(x0,y1),(xm,y1),(xm,y1-q),(xm,ym+q),(xm,ym),(xt,ym),(xm,ym),(xm,ym-q),(xm,y0+q),(xm,y0),(x0,y0)]
    c=[MPath.MOVETO,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CURVE3,MPath.CURVE3,MPath.CURVE3,MPath.CURVE3,MPath.LINETO,MPath.CURVE3,MPath.CURVE3]
    ax.add_patch(PathPatch(MPath(v,c),fill=False,edgecolor=INK,lw=0.6,zorder=4))
    return (xt,ym)
tip=brace(a,65.0,0.616,0.709,67.6,70.3,0.012)
an=a.annotate("small",xy=tip,xytext=(83.1,0.590),fontsize=6.5,ha="center",va="center",color=INK,zorder=5,
              arrowprops=dict(arrowstyle="-",lw=0.45,color=LEAD,shrinkA=1.2,shrinkB=0.5))
an.arrow_patch.set_zorder(2)
# panel (b)
lab(b,"SQuAD",[(15.4,0.811)],(6.703,0.828))
lab(b,"stronger helper",[(13.4,0.800),(21.2,0.792)],(9.498,0.733))
lab(b,"OLMo receiver",[(P(172,742),0.790),(P(67,299),0.749)],(54.26,0.781))
lab(b,"OLMo/\nMMLU-Pro",[(P(1112,2641),0.633)],(68.0,0.700))
lab(b,"Llama-3.2-1B helper",[(P(191,742),0.693),(P(88,299),0.678)],(18.0,0.638))
lab(b,"Llama-3.2-1B helper",[((P(427,742)+P(174,299))/2,0.6165)],(22.7,0.590))   # the two C2C points overlap
# zoom inset: Text+fact and the two Llama-8B settings lie too close to separate at this scale
zx=[p for p in ext if p[0] in ("fact","L8B")]
axins=b.inset_axes([0.50,0.655,0.49,0.325])
axins.set_xscale("log"); axins.set_xlim(9.4,13.2); axins.set_ylim(0.861,0.895)
axins.set_facecolor(BAND); axins.minorticks_off(); axins.set_xticks([]); axins.set_yticks([])
for sp in axins.spines.values(): sp.set_linewidth(0.5); sp.set_color(MUTED)
for g,pth,x,y,d in zx:
    axins.scatter([x],[y],s=26,marker="o",facecolor=TEXT,edgecolor=TEXT,lw=1.1,zorder=3)
kw=dict(fontsize=6.5,color=INK,va="center")
axins.text(10.2*1.028,0.889,"Text+fact",ha="left",**kw)
axins.text(P(37,299)/1.028,0.880,"Llama-8B/ARC",ha="right",**kw)
axins.text(9.55,0.8655,"Llama-8B/OBQA",ha="left",**kw)
from matplotlib.patches import Rectangle, ConnectionPatch
b.add_patch(Rectangle((9.4,0.861),13.2-9.4,0.895-0.861,fill=False,edgecolor=MUTED,lw=0.5,zorder=4))
for yd,ya in [(0.895,1.0),(0.861,0.0)]:
    b.add_artist(ConnectionPatch(xyA=(13.2,yd),coordsA=b.transData,xyB=(0.0,ya),coordsB=axins.transAxes,color=MUTED,lw=0.45,zorder=4))
draw_leaders()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "figures" / "figure2_boundary.pdf"
fig.savefig(OUT)
print("wrote", OUT)
