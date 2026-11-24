"""Run from the alzheimer/ folder."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','axes.labelcolor':'#333','xtick.color':'#444','ytick.color':'#333'})
S=json.load(open('results/metrics_all_models.json'))
HYB="HED-XAI-AD (VGG16–InceptionV3 + CA)"; V2="HED-XAI-AD-V2 (VGG16–DenseNet121 + CBAM)"
short={HYB:"HED-XAI-AD (proposed)",V2:"HED-XAI-AD-V2"}
nm=lambda k: short.get(k,k)
C1,C2,C3='#2a78d6','#eb6834','#1baf7a'
# ---------- confusion matrices ----------
order=[HYB,"VGG16-50","InceptionV3-50","VGG16-20",V2,"MobileNetV2 (B)-50","CNN-50","MobileNetV2 (A)-30"]
cmap=LinearSegmentedColormap.from_list('b',['#f4f8fd','#9cc3ee','#2a78d6','#0d3f7a'])
fig,axs=plt.subplots(2,4,figsize=(13,6.6),dpi=220)
for ax,k in zip(axs.flat,order):
    TN,FP,FN,TP=S[k]['cm']; m=np.array([[TN,FP],[FN,TP]])
    ax.imshow(m,cmap=cmap,vmin=0,vmax=300)
    for i in range(2):
        for j in range(2):
            v=m[i,j]; col='white' if v>170 else '#1a1a1a'
            ax.text(j,i-0.08,str(v),ha='center',va='center',fontsize=15,weight='bold',color=col)
            ax.text(j,i+0.22,f"{v/3:.1f}%",ha='center',va='center',fontsize=8.5,color=col)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['ND','VMD'],fontsize=9); ax.set_yticklabels(['ND','VMD'],fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    t=nm(k); ax.set_title(f"{t}\nAcc {S[k]['acc']*100:.2f}% · errors {FP+FN}",fontsize=9.6,weight='bold' if k==HYB else 'normal',color='#1F3864' if k==HYB else '#222')
    if k==HYB:
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(C2); s.set_linewidth(2.5)
for ax in axs[:,0]: ax.set_ylabel('Actual class',fontsize=9)
for ax in axs[1,:]: ax.set_xlabel('Predicted class',fontsize=9)
fig.text(0.5,0.005,'ND = Non Demented (n = 300), VMD = Very mild Dementia (n = 300). Cell percentages are relative to the 300 images of the actual class.',ha='center',fontsize=8.5,color='#555')
plt.tight_layout(rect=(0,0.03,1,1)); plt.subplots_adjust(hspace=0.42); plt.savefig('results/figures/dissertation/ad_confusion_panel.png',facecolor='white'); plt.close()
# ---------- metric comparison ----------
models=[k for k in sorted(S,key=lambda k:S[k]['acc'])]
fig,ax=plt.subplots(figsize=(11,6.4),dpi=220)
y=np.arange(len(models)); h=0.26
acc=[S[k]['acc']*100 for k in models]; lo=[S[k]['acc']*100-S[k]['lo']*100 for k in models]; hi=[S[k]['hi']*100-S[k]['acc']*100 for k in models]
ax.barh(y+h,[S[k]['sens']*100 for k in models],h*0.9,color=C2,label='Sensitivity')
ax.barh(y,acc,h*0.9,color=C1,label='Accuracy (with Wilson 95% CI)',xerr=[lo,hi],error_kw=dict(ecolor='#1a1a1a',elinewidth=1,capsize=2.5))
ax.barh(y-h,[S[k]['spec']*100 for k in models],h*0.9,color=C3,label='Specificity')
for yi,k in zip(y,models):
    ax.text(101.2,yi,f"{S[k]['acc']*100:.2f}%   MCC {S[k]['mcc']:.3f}",va='center',fontsize=8.6,color='#1F3864' if k==HYB else '#333',weight='bold' if k==HYB else 'normal')
ax.set_yticks(y); ax.set_yticklabels([nm(k) for k in models],fontsize=9.5)
for lbl,k in zip(ax.get_yticklabels(),models):
    if k==HYB: lbl.set_weight('bold'); lbl.set_color('#1F3864')
ax.set_xlim(60,100); ax.set_xlabel('Percentage (%) on the held-out test set (n = 600)',fontsize=9.5)
ax.xaxis.grid(True,color='#e6e6e6',lw=0.8); ax.set_axisbelow(True)
ax.legend(loc='lower center',bbox_to_anchor=(0.42,1.0),ncol=3,frameon=False,fontsize=9)
ax.axvline(90,color='#bbbbbb',lw=0.8,ls='--')
plt.tight_layout(); plt.subplots_adjust(right=0.80); plt.savefig('results/figures/dissertation/ad_metric_comparison.png',facecolor='white'); plt.close()
# ---------- trade-off ----------
pts=[k for k in S if S[k].get('ms')]
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
OFF={('ms','CNN-20'):(6,-12),('ms','MobileNetV2 (A)-30'):(-6,6),('size','InceptionV3-50'):(-6,4),('size','VGG16-50'):(6,4),('size','VGG16-20'):(6,-3),('size','HED-XAI-AD-V2'):(-6,6),('size','CNN-20'):(6,-12),('size','CNN-50'):(6,4),('size',HYB):(-6,6),('ms',HYB):(6,4)}
fig,axs=plt.subplots(1,2,figsize=(12,4.8),dpi=220)
for ax,key,xl in [(axs[0],'ms','Inference time (ms per image, log scale)'),(axs[1],'size','Saved model size (MB, log scale)')]:
    for k in pts:
        c=C2 if k==HYB else C1
        ax.scatter(S[k][key],S[k]['acc']*100,s=70 if k==HYB else 46,color=c,edgecolor='white',linewidth=1.5,zorder=3)
        dx,dy,ha=(1.08,0.35,'left')
        off=OFF.get((key,k),(6,4)); ha='right' if off[0]<0 else 'left'
        ax.annotate(nm(k),(S[k][key],S[k]['acc']*100),xytext=off,textcoords='offset points',ha=ha,fontsize=8.2,color='#1F3864' if k==HYB else '#333',weight='bold' if k==HYB else 'normal')
    ax.set_xscale('log'); ax.set_xlabel(xl,fontsize=9.5); ax.set_ylabel('Test accuracy (%)',fontsize=9.5)
    ax.grid(True,color='#ececec',lw=0.8); ax.set_axisbelow(True); ax.set_ylim(79,95)
axs[0].set_xlim(10,600); axs[1].set_xlim(15,400)
for ax,t in [(axs[0],[10,20,50,100,200,500]),(axs[1],[20,50,100,200,400])]:
    ax.xaxis.set_major_locator(FixedLocator(t)); ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in t])); ax.xaxis.set_minor_locator(NullLocator())
axs[0].set_title('(a) Accuracy vs. inference time',fontsize=10.5,loc='left',weight='bold',color='#1F3864')
axs[1].set_title('(b) Accuracy vs. model size',fontsize=10.5,loc='left',weight='bold',color='#1F3864')
plt.tight_layout(); plt.savefig('results/figures/dissertation/ad_tradeoff.png',facecolor='white'); plt.close()
print('done')
