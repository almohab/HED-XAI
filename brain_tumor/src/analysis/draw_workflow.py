"""Run from the brain_tumor/ folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Sans'
NAVY='#1F3864'
fig,ax=plt.subplots(figsize=(13,7.6),dpi=220); ax.set_xlim(0,130); ax.set_ylim(0,76); ax.axis('off')
def box(x,y,w,h,num,title,body,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.3",fc=fc,ec=ec,lw=1.4))
    ax.add_patch(plt.Circle((x+2.6,y+h-2.7),1.9,color=ec,zorder=3))
    ax.text(x+2.6,y+h-2.75,str(num),ha='center',va='center',color='white',fontsize=8.5,weight='bold',zorder=4)
    ax.text(x+5.2,y+h-2.7,title,ha='left',va='center',fontsize=9.2,weight='bold',color='#111')
    ax.text(x+1.6,y+h-5.6,body,ha='left',va='top',fontsize=7.7,color='#222',linespacing=1.32)
def arr(x1,y1,x2,y2,cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,color=NAVY,lw=1.6,connectionstyle=cs))
ax.text(65,74,"Experimental workflow of the brain tumour study",ha='center',fontsize=13,weight='bold',color=NAVY)
# lane labels
for y,lab,c in [(51,'DATA',"#2E75B6"),(27,'MODELLING',"#548235"),(3,'EVALUATION &\nEXPLANATION',"#C55A11")]:
    ax.add_patch(FancyBboxPatch((0.5,y),7,19,boxstyle="round,pad=0.2,rounding_size=1",fc=c,ec=c))
    ax.text(4,y+9.5,lab,rotation=90,ha='center',va='center',color='white',fontsize=8.5,weight='bold')
B='#DEEBF7'; G='#E2EFDA'; O='#FBE5D6'
# row 1
box(10,51,27,19,1,'Source data','Figshare #1512427: 3,064 contrast-\nenhanced T1-weighted slices from\n233 patients; glioma 1,426,\nmeningioma 708, pituitary 930',B,'#2E75B6')
box(41,51,27,19,2,'Format conversion','MATLAB .mat → 8-bit JPEG;\nper-image min–max scaling\nto 0–255; matrix transposed\nto correct orientation',B,'#2E75B6')
box(72,51,27,19,3,'Per-class split','Random 70/15/15 within each\nclass (Python seed 42)\nTrain 2,144 · Val 458 · Test 462\n(test: 215 / 107 / 140)',B,'#2E75B6')
box(103,51,26,19,4,'Integrity audit','SHA-256 of every file:\n0 cross-split duplicates;\npatient IDs not used\n(image-level split)',B,'#2E75B6')
for a,b in [(37,41),(68,72),(99,103)]: arr(a+0.4,60.5,b-0.4,60.5)
# row 2
ax.add_patch(FancyBboxPatch((49,27),80,19,boxstyle="round,pad=0.3,rounding_size=1.3",fc=G,ec='#548235',lw=1.4))
ax.add_patch(plt.Circle((51.6,43.3),1.9,color='#548235',zorder=3)); ax.text(51.6,43.25,'5',ha='center',va='center',color='white',fontsize=8.5,weight='bold',zorder=4)
ax.text(54.2,43.3,'Candidate models — all trained on the identical split',fontsize=9.2,weight='bold',va='center')
ax.text(50.6,40.2,'Benchmarks (single backbone)',fontsize=8,weight='bold',color='#385723',va='top')
ax.text(50.6,37.2,'Custom CNN (50 / 100 epochs), from scratch\nVGG16 (10 / 20), DenseNet201 (10 / 20)\nMobileNetV2 (10 / 20), InceptionV3 (10 / 20)\nfrozen ImageNet backbones, /255 input',fontsize=7.7,va='top',linespacing=1.32)
ax.plot([89,89],[29,40.5],color='#548235',lw=0.8,ls=':')
ax.text(90.6,40.2,'Proposed model (lightweight hybrid)',fontsize=8,weight='bold',color='#385723',va='top')
ax.text(90.6,37.2,'HED-XAI-BT: MobileNetV2 three-scale\n  fusion + channel attention + softmax\n50 epochs (100-epoch confirmation run),\n  last 30 layers fine-tuned, BN frozen',fontsize=7.7,va='top',linespacing=1.32)
box(10,27,35,19,6,'Model selection','HED-XAI-BT: best val_loss\ncheckpoint (epoch 11);\nbaselines: EarlyStopping\n(patience 5) or fixed epochs;\ntest set used once at the end',G,'#548235')
arr(116,50.6,116,46.4)
arr(48.6,36.5,45.4,36.5)
# row 3
box(10,3,35,19,7,'Held-out test (n = 462)','Accuracy (Wilson 95% CI), per-class\nrecall, precision and specificity,\nmacro-F1, multi-class MCC, κ,\none-vs-rest ROC-AUC, confusion matrix',O,'#C55A11')
box(49,3,38,19,8,'Efficiency profiling','Parameter count, saved model size (MB),\ninference time (ms/image), throughput,\ntraining time; trade-off against\naccuracy',O,'#C55A11')
box(91,3,38,19,9,'Grad-CAM explanation','VGG16-20 benchmark, block5_conv3\n(14×14); heatmap + JET overlay;\nlesion-localisation review per class\n(glioma, meningioma, pituitary)',O,'#C55A11')
arr(27,26.6,27,22.4); arr(45.4,12.5,48.6,12.5); arr(87.4,12.5,90.6,12.5)
plt.savefig('results/figures/dissertation/bt_workflow.png',bbox_inches='tight',facecolor='white')
