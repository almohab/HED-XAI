"""Run from the skin_cancer/ folder (analysis of the recorded results; paths may need adjusting)."""
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
sh=lambda im,h: im.resize((round(im.width*h/im.height),h),Image.LANCZOS)
o=lambda p: Image.open(W+p).convert('RGB')
ims=[lab(sh(o(f'sc/177b60d9/{f}'),560),t) for f,t in [('c001_12.png','(a) ROC curve'),('c001_13.png','(b) Precision–recall curve')]]
hstack(ims,30).save(W+'scfig/sc_hybrid_eval.png')
g=[o('sc/e1bb4c32/c012_0.png')]+[Image.open(A+f).convert('RGB') for f in ['b200578f-image.png','4e17de7d-image.png','af1a7543-image.png']]
t=['(a) Benign test lesion (700.jpg), predicted benign, p(malignant) = 0.013',
   '(b) Supplied example: large irregularly pigmented lesion',
   '(c) Supplied example: small dark lesion with hairs',
   '(d) Supplied example: homogeneous brown lesion']
vstack([lab(i.resize((1480,510)),s) for i,s in zip(g,t)],14).save(W+'scfig/sc_gradcam.png')
for f in ['sc_hybrid_eval','sc_gradcam']: print(f, Image.open(W+'scfig/'+f+'.png').size)
