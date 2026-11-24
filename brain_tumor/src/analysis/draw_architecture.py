"""Run from the brain_tumor/ folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Sans'
NAVY='#1F3864'; BLUE='#DCE6F2'; GREEN='#E2EFDA'; YEL='#FFF2CC'; PURP='#E4DFEC'; GREY='#F2F2F2'
fig,ax=plt.subplots(figsize=(13.5,6.4),dpi=220); ax.set_xlim(0,136); ax.set_ylim(0,64); ax.axis('off')
def box(x,y,w,h,title,sub=None,fc=BLUE,ec=NAVY,fs=8.6,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=1.2",fc=fc,ec=ec,lw=1.3,ls=ls))
    if sub:
        ax.text(x+w/2,y+h*0.66,title,ha='center',va='center',fontsize=fs,weight='bold',color='#111')
        ax.text(x+w/2,y+h*0.30,sub,ha='center',va='center',fontsize=fs-1.6,color='#333',linespacing=1.15)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,weight='bold',color='#111',linespacing=1.15)
def arr(x1,y1,x2,y2,c=NAVY,ls='-',cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color=c,lw=1.4,ls=ls,connectionstyle=cs))
ax.text(68,62,'Lightweight HED-XAI: MobileNetV2 multi-scale fusion with channel attention (TB, PN, BT)',ha='center',fontsize=12.5,weight='bold',color=NAVY)
box(1,24,11,14,'Input','224×224×3\nMobileNetV2\npreprocess\n[−1, 1]',fc=GREY)
# backbone column
ax.add_patch(FancyBboxPatch((15,9),26,48,boxstyle="round,pad=0.3,rounding_size=1.5",fc='#F5F8FC',ec='#7F9DC6',lw=1,ls='--'))
ax.text(28,55,'MobileNetV2 backbone (ImageNet)',ha='center',fontsize=8.6,weight='bold',color='#2F5597')
ax.text(28,52.2,'last 30 layers fine-tuned, BatchNorm frozen',ha='center',fontsize=7.3,color='#333')
box(17,40,22,9,'block_3_expand_relu','56×56×144 — edges, texture',fs=8)
box(17,26,22,9,'block_6_expand_relu','28×28×192 — local structure',fs=8)
box(17,12,22,9,'out_relu','7×7×1280 — semantics',fs=8)
arr(12,31,14.7,31)
arr(28,39.7,28,35.3); arr(28,25.7,28,21.3)
for y,d in [(44.5,'144'),(30.5,'192'),(16.5,'1,280')]:
    box(44,y-3.5,8,7,'GAP',f'{d}-d',fc=BLUE,fs=8)
    arr(39.3,y,43.7,y)
box(56,22,10,20,'Concat-\nenate','144+192\n+1,280\n= 1,616-d',fc=GREEN)
for y in (44.5,30.5,16.5): arr(52.3,y,55.7,32,cs="arc3,rad=0")
box(69,22,12,20,'Fusion','BatchNorm\n→ Dense 256\n(ReLU)\n→ Dropout 0.35',fc=GREEN)
arr(66.3,32,68.7,32)
ax.add_patch(FancyBboxPatch((84,18.5),18,27,boxstyle="round,pad=0.3,rounding_size=1.5",fc='#FAF7FD',ec='#7030A0',lw=1.2))
ax.text(93,43.4,'Channel attention',ha='center',fontsize=8.6,weight='bold',color='#7030A0')
box(85.5,34,15,5.8,'Dense 32, ReLU',fc=PURP,fs=7.4)
box(85.5,26,15,5.8,'Dense 256, sigmoid',fc=PURP,fs=7.4)
arr(93,33.8,93,32.1)
ax.text(93,21.3,'f′ = f ⊙ σ(W₂δ(W₁f))',ha='center',fontsize=7.6,style='italic',color='#333')
arr(81.3,32,83.7,32)
box(105,22,11,20,'Head','Dense 128\n(ReLU)\nDropout 0.25',fc=YEL)
arr(102.3,32,104.7,32)
box(119,22,16,20,'Output','TB, PN: 1 sigmoid\nBT: 3-way softmax\n(glioma,\nmeningioma,\npituitary)',fc='#C00000',ec='#7F0000',fs=8.4)
ax.texts[-2].set_color('white'); ax.texts[-1].set_color('white')
arr(116.3,32,118.7,32)
box(43,0.8,24,9.6,'Grad-CAM target','out_relu (7×7); gradients flow\nback through the pooled head',fc='white',ec='#C00000',fs=8,ls='--')
arr(28,11.7,42.7,6,c='#C00000',ls='--',cs="arc3,rad=0.3")
ax.add_patch(FancyBboxPatch((70,2.5),65,12.5,boxstyle="round,pad=0.3,rounding_size=1.2",fc='#F7F9FC',ec='#A6A6A6',lw=0.9))
ax.text(71.5,12.6,'Training (identical for TB, PN and BT)',fontsize=8.4,weight='bold',color=NAVY)
ax.text(71.5,7.6,'Adam 1×10⁻⁴, batch 16, ModelCheckpoint (best val_loss),\nReduceLROnPlateau (×0.2, patience 2, min 1×10⁻⁷).',fontsize=7.5)
ax.text(71.5,4.0,'Cross-entropy loss (sparse categorical for BT); light geometric augmentation; seed 42.',fontsize=7.5)
ax.text(135,0.0,'≈2.73 M parameters (2,728,355 for HED-XAI-BT; 1,977,859 trainable); saved model ≈26 MB',ha='right',fontsize=7.4,color='#555',style='italic')
plt.savefig('results/figures/dissertation/hedxai_lite_architecture.png',bbox_inches='tight',facecolor='white')
