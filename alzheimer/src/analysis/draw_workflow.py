"""Run from the alzheimer/ folder."""
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
ax.text(65,74,"Experimental workflow of the Alzheimer's disease study",ha='center',fontsize=13,weight='bold',color=NAVY)
# lane labels
for y,lab,c in [(51,'DATA',"#2E75B6"),(27,'MODELLING',"#548235"),(3,'EVALUATION &\nEXPLANATION',"#C55A11")]:
    ax.add_patch(FancyBboxPatch((0.5,y),7,19,boxstyle="round,pad=0.2,rounding_size=1",fc=c,ec=c))
    ax.text(4,y+9.5,lab,rotation=90,ha='center',va='center',color='white',fontsize=8.5,weight='bold')
B='#DEEBF7'; G='#E2EFDA'; O='#FBE5D6'
# row 1
box(10,51,27,19,1,'Source data','Kaggle distribution of OASIS-1\ncross-sectional T1-weighted MRI\nslices; classes Non Demented and\nVery mild Dementia (CDR 0 vs 0.5)',B,'#2E75B6')
box(41,51,27,19,2,'Balanced sampling','2,000 slices drawn at random\nper class → 4,000 images\n(1:1 class ratio, no\noversampling required)',B,'#2E75B6')
box(72,51,27,19,3,'Integrity audit','SHA-256 hash of every file;\nexact duplicates removed: 0\nsplit manifest saved to CSV\nfor reproducibility',B,'#2E75B6')
box(103,51,26,19,4,'Stratified split','70 / 15 / 15, seed 42\nTrain 2,800 (1,400 + 1,400)\nVal 600 (300 + 300)\nTest 600 (300 + 300)',B,'#2E75B6')
for a,b in [(37,41),(68,72),(99,103)]: arr(a+0.4,60.5,b-0.4,60.5)
# row 2
ax.add_patch(FancyBboxPatch((49,27),80,19,boxstyle="round,pad=0.3,rounding_size=1.3",fc=G,ec='#548235',lw=1.4))
ax.add_patch(plt.Circle((51.6,43.3),1.9,color='#548235',zorder=3)); ax.text(51.6,43.25,'5',ha='center',va='center',color='white',fontsize=8.5,weight='bold',zorder=4)
ax.text(54.2,43.3,'Candidate models — all trained on the identical split',fontsize=9.2,weight='bold',va='center')
ax.text(50.6,40.2,'Benchmarks (single backbone)',fontsize=8,weight='bold',color='#385723',va='top')
ax.text(50.6,37.2,'Custom CNN (20 / 50 epochs)\nVGG16, Flatten head (10 / 20 / 50 epochs)\nInceptionV3, 299×299 (50 epochs)\nMobileNetV2 (A: frozen + fine-tune; B: ES)',fontsize=7.7,va='top',linespacing=1.32)
ax.plot([89,89],[29,40.5],color='#548235',lw=0.8,ls=':')
ax.text(90.6,40.2,'Proposed hybrids (dual backbone)',fontsize=8,weight='bold',color='#385723',va='top')
ax.text(90.6,37.2,'HED-XAI-AD-V2 (design iteration):\n  VGG16 + DenseNet121, multi-scale, CBAM\nHED-XAI-AD (final): VGG16 + InceptionV3,\n  GAP fusion, channel attention, 2-stage',fontsize=7.7,va='top',linespacing=1.32)
box(10,27,35,19,6,'Model selection','Checkpoint chosen on validation\nloss only (ModelCheckpoint);\nReduceLROnPlateau and EarlyStopping\nmonitor validation data only; the\ntest partition is never consulted',G,'#548235')
arr(116,50.6,116,46.4)
arr(48.6,36.5,45.4,36.5)
# row 3
box(10,3,35,19,7,'Held-out test (n = 600)','Accuracy (Wilson 95% CI), sensitivity,\nspecificity, precision, NPV, F1, MCC,\nCohen\'s κ, ROC-AUC, PR-AUC,\nconfusion matrix (threshold τ = 0.5)',O,'#C55A11')
box(49,3,38,19,8,'Efficiency profiling','Parameter count, saved model size (MB),\ninference time (ms/image), throughput,\ntraining time; trade-off against\naccuracy',O,'#C55A11')
box(91,3,38,19,9,'Grad-CAM explanation','Target layer block5_conv3 (14×14);\nheatmap + JET overlay (0.6 / 0.4);\nqualitative neuroanatomical review\n(periventricular, temporal, parietal)',O,'#C55A11')
arr(27,26.6,27,22.4); arr(45.4,12.5,48.6,12.5); arr(87.4,12.5,90.6,12.5)
plt.savefig('results/figures/dissertation/ad_workflow.png',bbox_inches='tight',facecolor='white')
