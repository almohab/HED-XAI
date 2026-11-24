"""Run from the brain_tumor/ folder."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a9a','axes.labelcolor':'#333','xtick.color':'#444','ytick.color':'#333'})
S=json.load(open('results/metrics_all_models.json'))
HYB="HED-XAI-BT (proposed)"; OUT='results/figures/dissertation/'
C1,C2,C3='#2a78d6','#eb6834','#1baf7a'
cmap=LinearSegmentedColormap.from_list('b',['#f4f8fd','#9cc3ee','#2a78d6','#0d3f7a'])
lab=['Gli','Men','Pit']; sup=[215,107,140]
order=["HED-XAI-BT (proposed)","CNN-100","VGG16-20","CNN-50","DenseNet201-20","MobileNetV2-20","InceptionV3-20","MobileNetV2-10"]
fig,axs=plt.subplots(2,4,figsize=(13,6.9),dpi=220)
for ax,k in zip(axs.flat,order):
    m=np.array(S[k]['cm']); rown=m/m.sum(1,keepdims=True)
    ax.imshow(rown,cmap=cmap,vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):
            v=m[i,j]; col='white' if rown[i,j]>0.55 else '#1a1a1a'
            ax.text(j,i-0.1,str(v),ha='center',va='center',fontsize=12.5,weight='bold',color=col)
            ax.text(j,i+0.24,f"{rown[i,j]*100:.1f}%",ha='center',va='center',fontsize=7.4,color=col)
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(lab,fontsize=9); ax.set_yticklabels(lab,fontsize=9)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"{k}\nAcc {S[k]['acc']*100:.2f}% · errors {S[k]['err']}",fontsize=9.6,weight='bold' if k==HYB else 'normal',color='#1F3864' if k==HYB else '#222')
    if k==HYB:
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(C2); s.set_linewidth(2.5)
for ax in axs[:,0]: ax.set_ylabel('Actual class',fontsize=9)
for ax in axs[1,:]: ax.set_xlabel('Predicted class',fontsize=9)
fig.text(0.5,0.005,'Gli = glioma (n = 215), Men = meningioma (n = 107), Pit = pituitary (n = 140). Percentages are row-normalised (recall per actual class).',ha='center',fontsize=8.5,color='#555')
plt.tight_layout(rect=(0,0.03,1,1)); plt.subplots_adjust(hspace=0.45); plt.savefig(OUT+'bt_confusion_panel.png',facecolor='white'); plt.close()

# per-class recall grouped bars + accuracy CI
models=sorted(S,key=lambda k:S[k]['acc'])
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,6.6),dpi=220,gridspec_kw={'width_ratios':[1.35,1]})
y=np.arange(len(models)); h=0.26
for j,(c,nm) in enumerate([(C1,'Glioma'),(C2,'Meningioma'),(C3,'Pituitary')]):
    a1.barh(y+(1-j)*h,[S[k]['rec'][j]*100 for k in models],h*0.9,color=c,label=f'{nm} recall')
a1.set_yticks(y); a1.set_yticklabels(models,fontsize=9.2)
for lbl,k in zip(a1.get_yticklabels(),models):
    if k==HYB: lbl.set_weight('bold'); lbl.set_color('#1F3864')
a1.set_xlim(70,100); a1.set_xlabel('Per-class recall (%)',fontsize=9.5); a1.xaxis.grid(True,color='#e6e6e6'); a1.set_axisbelow(True)
a1.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,frameon=False,fontsize=8.8)
a1.set_title('(a) Recall by tumour class',loc='left',fontsize=10.5,weight='bold',color='#1F3864',pad=26)
acc=[S[k]['acc']*100 for k in models]; lo=[S[k]['acc']*100-S[k]['lo']*100 for k in models]; hi=[S[k]['hi']*100-S[k]['acc']*100 for k in models]
cols=[C2 if k==HYB else C1 for k in models]
a2.errorbar(acc,y,xerr=[lo,hi],fmt='none',ecolor='#555',elinewidth=1.2,capsize=3,zorder=2)
a2.scatter(acc,y,color=cols,s=[70 if k==HYB else 45 for k in models],zorder=3,edgecolor='white',linewidth=1.2)
for yi,k in zip(y,models): a2.text(S[k]['hi']*100+0.25,yi,f"{S[k]['acc']*100:.2f}%  (MCC {S[k]['mcc']:.3f})",va='center',fontsize=8.2,color='#1F3864' if k==HYB else '#333',weight='bold' if k==HYB else 'normal')
a2.set_yticks(y); a2.set_yticklabels([]); a2.set_xlim(84,101.5); a2.set_xlabel('Accuracy (%) with Wilson 95% CI',fontsize=9.5)
a2.xaxis.grid(True,color='#e6e6e6'); a2.set_axisbelow(True)
a2.set_title('(b) Overall accuracy',loc='left',fontsize=10.5,weight='bold',color='#1F3864',pad=26)
plt.tight_layout(); plt.savefig(OUT+'bt_metric_comparison.png',facecolor='white'); plt.close()

# trade-off: accuracy vs size (sizes known for all); single panel
fig,ax=plt.subplots(figsize=(8.6,4.9),dpi=220)
OFF={"CNN-100":(6,4),"CNN-50":(6,4),"VGG16-20":(-6,4),"VGG16-10":(-6,-11),"DenseNet201-20":(6,4),"DenseNet201-10":(6,-11),"InceptionV3-20":(6,4),"InceptionV3-10":(6,-10),"MobileNetV2-20":(6,-11),"MobileNetV2-10":(6,4),HYB:(6,5)}
for k,d in S.items():
    c=C2 if k==HYB else C1
    ax.scatter(d['size'],d['acc']*100,s=75 if k==HYB else 45,color=c,edgecolor='white',linewidth=1.3,zorder=3)
    o=OFF.get(k,(6,4)); ax.annotate(k,(d['size'],d['acc']*100),xytext=o,textcoords='offset points',ha='right' if o[0]<0 else 'left',fontsize=8.2,color='#1F3864' if k==HYB else '#333',weight='bold' if k==HYB else 'normal')
ax.set_xscale('log'); ax.set_xlim(10,400)
ax.xaxis.set_major_locator(FixedLocator([10,20,50,100,200,400])); ax.xaxis.set_major_formatter(FixedFormatter(['10','20','50','100','200','400'])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel('Saved model size (MB, log scale)',fontsize=9.5); ax.set_ylabel('Test accuracy (%)',fontsize=9.5); ax.set_ylim(87,96)
ax.grid(True,color='#ececec'); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT+'bt_tradeoff.png',facecolor='white'); plt.close()
print('ok')
