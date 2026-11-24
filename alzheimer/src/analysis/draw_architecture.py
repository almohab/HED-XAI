"""Run from the alzheimer/ folder."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Sans'
NAVY='#1F3864'; BLUE='#DCE6F2'; GREEN='#E2EFDA'; ORANGE='#FBE5D6'; YEL='#FFF2CC'; PURP='#E4DFEC'; GREY='#F2F2F2'
fig,ax=plt.subplots(figsize=(13.5,6.6),dpi=220); ax.set_xlim(0,136); ax.set_ylim(0,64); ax.axis('off')
def box(x,y,w,h,title,sub=None,fc=BLUE,ec=NAVY,fs=8.6,bold=True,lw=1.3,ls='-'):
    p=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.25,rounding_size=1.2",fc=fc,ec=ec,lw=lw,ls=ls); ax.add_patch(p)
    if sub:
        ax.text(x+w/2,y+h*0.64,title,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',color='#111')
        ax.text(x+w/2,y+h*0.29,sub,ha='center',va='center',fontsize=fs-1.6,color='#333',linespacing=1.15)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',color='#111',linespacing=1.15)
def arr(x1,y1,x2,y2,c=NAVY,ls='-',lw=1.4,cs="arc3"):
    a=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color=c,lw=lw,ls=ls,connectionstyle=cs); ax.add_patch(a)
ax.text(68,62,'HED-XAI-AD: dual-backbone hybrid with channel attention',ha='center',fontsize=13,weight='bold',color=NAVY)
# input
box(1,25,11,14,'Input','T1-weighted\nMRI slice\n224×224×3\n(raw 0–255)',fc=GREY)
# branch labels
ax.add_patch(FancyBboxPatch((15,38.5),50,20,boxstyle="round,pad=0.3,rounding_size=1.5",fc='none',ec='#7F9DC6',lw=1,ls='--'))
ax.text(16,57,'Branch A — VGG16 (local texture, fine cortical detail)',fontsize=8.2,color='#2F5597',weight='bold')
ax.add_patch(FancyBboxPatch((15,5.5),50,20,boxstyle="round,pad=0.3,rounding_size=1.5",fc='none',ec='#C55A11',lw=1,ls='--'))
ax.text(16,7.0,'Branch B — InceptionV3 (multi-scale receptive fields)',fontsize=8.2,color='#C55A11',weight='bold')
# branch A
box(16,42,11,12,'VGG16\npreprocess','RGB→BGR,\nmean subtr.',fc=BLUE,fs=8.2)
box(30,42,17,12,'VGG16','ImageNet weights\nblock5_conv1–3\nfine-tuned (Stage 2)',fc=BLUE)
box(49,42,7,12,'7×7×\n512',fc='white',fs=8)
box(58.5,42,6,12,'GAP',fc=BLUE,fs=8.5)
# branch B
box(16,10,11,12,'Resize\n299×299','+ scale to\n[−1, 1]',fc=ORANGE,fs=8.2)
box(30,10,17,12,'InceptionV3','ImageNet weights\nlast 30 layers\nfine-tuned, BN frozen',fc=ORANGE)
box(49,10,7,12,'8×8×\n2048',fc='white',fs=8)
box(58.5,10,6,12,'GAP',fc=ORANGE,fs=8.5)
arr(12,34,15.5,47); arr(12,30,15.5,17)
for (a,b) in [(27,30),(47,49),(56,58.5)]:
    arr(a+0.3,48,b-0.3,48); arr(a+0.3,16,b-0.3,16)
# fusion
box(68,24,11,16,'Concat-\nenate','512 + 2048\n= 2,560-d',fc=GREEN)
arr(64.8,48,72,40.4,cs="arc3,rad=-0.15"); arr(64.8,16,72,23.6,cs="arc3,rad=0.15")
box(81.5,24,12,16,'Fusion','BN → Dense 512\n(ReLU)\n→ Dropout 0.40',fc=GREEN)
arr(79.3,32,81.2,32)
# attention
ax.add_patch(FancyBboxPatch((96,19.5),18,25,boxstyle="round,pad=0.3,rounding_size=1.5",fc='#FAF7FD',ec='#7030A0',lw=1.2))
ax.text(105,42.2,'Channel attention',ha='center',fontsize=8.6,weight='bold',color='#7030A0')
box(97.5,33,15,5.8,'Dense 64, ReLU',fc=PURP,fs=7.6,bold=False)
box(97.5,25.5,15,5.8,'Dense 512, sigmoid',fc=PURP,fs=7.6,bold=False)
arr(105,32.8,105,31.5)
ax.text(105,21.6,'f′ = f ⊙ σ(W₂δ(W₁f))',ha='center',fontsize=7.6,color='#333',style='italic')
arr(93.8,32,95.7,32)
# head
box(116.5,24,10.5,16,'Head','Dense 256, BN\nDropout 0.35\nDense 128\nDropout 0.25',fc=YEL,fs=8.6)
arr(114.3,32,116.2,32)
box(128.5,26,7.3,12,'Output','sigmoid\nP(VMD)\nτ = 0.5',fc='#C00000',ec='#7F0000',fs=8.6)
ax.texts[-2].set_color('white'); ax.texts[-1].set_color('white')
arr(127.3,32,128.2,32)
# grad-cam tap
box(70,48,30,9,'Grad-CAM-compatible tap','class-score gradients w.r.t. block5_conv3 (14×14)\n→ channel-weighted map → JET overlay',fc='white',ec='#C00000',fs=8,ls='--')
arr(42,54.3,69.7,52.5,c='#C00000',ls='--',cs="arc3,rad=-0.12")
# training strip
ax.add_patch(FancyBboxPatch((68,2.5),67.5,13.5,boxstyle="round,pad=0.3,rounding_size=1.2",fc='#F7F9FC',ec='#A6A6A6',lw=0.9))
ax.text(69.5,13.6,'Two-stage optimisation',fontsize=8.4,weight='bold',color=NAVY)
ax.text(69.5,9.6,'Stage 1 (10 epochs): both backbones frozen; head trained, Adam lr 1×10⁻³',fontsize=7.6,color='#222')
ax.text(69.5,6.6,'Stage 2 (40 epochs): partial unfreezing, Adam lr 1×10⁻⁵, ReduceLROnPlateau (×0.2, p = 3),',fontsize=7.6,color='#222')
ax.text(69.5,4.1,'EarlyStopping (p = 8), best-val_loss checkpoint. Batch 8, binary cross-entropy, seed 42.',fontsize=7.6,color='#222')
ax.text(135.5,0.0,'VMD = Very mild Dementia. Total 38.07 M parameters (13.78 M trainable in Stage 2)',ha='right',fontsize=7.4,color='#555',style='italic')
plt.savefig('results/figures/dissertation/ad_architecture.png',bbox_inches='tight',facecolor='white')
