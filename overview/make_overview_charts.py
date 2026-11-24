"""Run from the repository root: python overview/make_overview_charts.py (values taken from the per-disease results)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','xtick.color':'#444','ytick.color':'#333'})
C1,C2='#2a78d6','#eb6834'
# (model, size MB, inference ms) — values from the efficiency table; None = not recorded
D={'Tuberculosis':[('HED-XAI-TB',25.89,10.86),('VGG16 (clean)',112.13,86.92),('DenseNet201-20',None,74.92)],
   'Pneumonia':[('HED-XAI-PN',26.10,14.84),('CNN-A-20 †',151.55,15.53),('InceptionV3-10',129.51,126.40),('DenseNet121-20',None,88.36),('MobileNetV2-20',None,50.44),('VGG16-20',None,274.88)],
   'Brain tumour':[('HED-XAI-BT',26.10,13.70),('MobileNetV2-20',14.14,None),('DenseNet201-20',83.25,None),('InceptionV3-20',96.14,None),('VGG16-20',203.25,260.4),('CNN-100 / -50',254.66,None)],
   "Alzheimer's disease":[('HED-XAI-AD',146.37,97.6),('HED-XAI-AD-V2',223.80,237.0),('MobileNetV2 (B)-50',22.78,23.3),('CNN-50',112.93,15.4),('InceptionV3-50',129.51,112.4),('VGG16-50',184.11,272.7)],
   'Skin cancer':[('HED-XAI-SC',26.10,18.14),('DenseNet121-10',36.16,None),('VGG16-GAP-20',111.74,None),('InceptionV3-10',129.51,136.93)]}
def panel(idx, unit, fname, title_unit, xmax):
    fig,axs=plt.subplots(2,3,figsize=(13,7.6),dpi=220)
    for ax,(dis,rows) in zip(axs.flat,D.items()):
        r=[(m,v[idx]) for m,*v in rows if v[idx] is not None]
        r=sorted(r,key=lambda t:t[1])
        y=np.arange(len(r))
        cols=[C2 if m.startswith('HED') else C1 for m,_ in r]
        ax.barh(y,[v for _,v in r],0.6,color=cols)
        for yi,(m,v) in zip(y,r):
            ax.text(v+xmax*0.015,yi,f"{v:.2f}" if v<100 else f"{v:.1f}",va='center',fontsize=8,color='#1F3864' if m.startswith('HED') else '#333',weight='bold' if m.startswith('HED') else 'normal')
        ax.set_yticks(y); ax.set_yticklabels([m for m,_ in r],fontsize=8.6)
        for l,(m,_) in zip(ax.get_yticklabels(),r):
            if m.startswith('HED'): l.set_weight('bold'); l.set_color('#1F3864')
        ax.set_xlim(0,xmax); ax.xaxis.grid(True,color='#ececec'); ax.set_axisbelow(True)
        ax.set_title(dis,loc='left',fontsize=10.5,weight='bold',color='#1F3864'); ax.set_xlabel(title_unit,fontsize=8.8)
    ax=axs.flat[5]; ax.axis('off')
    ax.text(0.02,0.85,'Proposed HED-XAI model',fontsize=9.5,color=C2,weight='bold',transform=ax.transAxes)
    ax.text(0.02,0.74,'Benchmark',fontsize=9.5,color=C1,weight='bold',transform=ax.transAxes)
    ax.text(0.02,0.58,"Only models with a recorded value are shown.\nValues were measured in the notebooks of\neach disease, partly on different computers\nand with different file formats, so they should\nbe compared within, not across, diseases.\n† CNN-A-20: earlier run.",fontsize=8.6,color='#444',va='top',transform=ax.transAxes,linespacing=1.4)
    plt.tight_layout(); plt.savefig('overview/'+fname,facecolor='white'); plt.close()
panel(0,'MB','global_model_size.png','Saved model size (MB)',300)
panel(1,'ms','global_inference_time.png','Inference time per image (ms)',320)
print('ok')
