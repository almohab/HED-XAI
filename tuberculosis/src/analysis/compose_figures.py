"""Run from the tuberculosis/ folder (analysis of the recorded results; paths may need adjusting)."""
import sys
from PIL import Image, ImageDraw, ImageFont
W='results/'; A='/mnt/user-data/attachments/00294aa0-4669-5e77-944b-0b0344daf709/'
F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
def lab(im,t,pad=38):
    c=Image.new('RGB',(im.width,im.height+pad),'white'); c.paste(im,(0,pad)); ImageDraw.Draw(c).text((10,6),t,fill='#1F3864',font=F); return c
def hstack(ims,gap=20):
    h=max(i.height for i in ims); w=sum(i.width for i in ims)+gap*(len(ims)-1); c=Image.new('RGB',(w,h),'white'); x=0
    for i in ims: c.paste(i,(x,(h-i.height)//2)); x+=i.width+gap
    return c
def vstack(ims,gap=24):
    w=max(i.width for i in ims); h=sum(i.height for i in ims)+gap*(len(ims)-1); c=Image.new('RGB',(w,h),'white'); y=0
    for i in ims: c.paste(i,((w-i.width)//2,y)); y+=i.height+gap
    return c
sw=lambda im,w: im.resize((w,round(im.height*w/im.width)),Image.LANCZOS)
sh=lambda im,h: im.resize((round(im.width*h/im.height),h),Image.LANCZOS)
o=lambda p: Image.open(W+p).convert('RGB')
# original HED-XAI-TB evaluation outputs (best checkpoint, Protocol B)
ims=[sh(o(f'tb/958c80ff/{f}'),560) for f in ['c001_4.png','c001_5.png']]
ims=[lab(i,t) for i,t in zip(ims,['(a) ROC curve','(b) Precision–recall curve'])]
hstack(ims,30).save(W+'tbfig/tb_hybrid_eval.png')
# original training curves of three Protocol A benchmarks
a=lab(sw(o('tb/51c48b7b/c001_3.png'),1300),'(a) VGG16-20')
b=lab(sw(o('tb/a219d358/c001_13.png'),1300),'(b) CNN-100')
c=lab(sw(o('tb/b409fbfe/c004_3.png'),1300),'(c) InceptionV3-20')
vstack([a,b,c],24).save(W+'tbfig/tb_protocolA_curves.png')
# Grad-CAM (VGG16-20, Protocol A model)
g=[Image.open(A+f).convert('RGB') for f in ['40f1fe92-image.png','92dbbde1-image.png']]
print([i.size for i in g])
g=[lab(i,t) for i,t in zip(g,['(a) Tuberculosis image (TB.54), predicted TB with p ≈ 1.000 (logit +71.0)','(b) Normal image (Normal-1860), predicted Normal with p ≈ 9×10⁻¹² (logit −25.4)'])]
vstack(g,16).save(W+'tbfig/tb_gradcam.png')
for f in ['tb_hybrid_eval','tb_protocolA_curves','tb_gradcam']: print(f, Image.open(W+'tbfig/'+f+'.png').size)
