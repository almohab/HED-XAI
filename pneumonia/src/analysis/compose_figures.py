"""Run from the pneumonia/ folder (analysis of the recorded results; paths may need adjusting)."""
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
sw=lambda im,w: im.resize((w,round(im.height*w/im.width)),Image.LANCZOS)
o=lambda p: Image.open(W+p).convert('RGB')
ims=[lab(sh(o(f'pn/f057bdce/{f}'),560),t) for f,t in [('c000_10.png','(a) ROC curves'),('c000_11.png','(b) Precision–recall curves')]]
hstack(ims,30).save(W+'pnfig/pn_hybrid_eval.png')
# original benchmark training curves
a=lab(sw(o('pn/5bdc9ced/c019_9.png'),1300),'(a) DenseNet121-20'); b=lab(sw(o('pn/5bdc9ced/c025_9.png'),1300),'(b) MobileNetV2-20')
c=lab(sw(hstack([o('pn/5bdc9ced/c028_10.png'),o('pn/5bdc9ced/c028_11.png')]),1300),'(c) InceptionV3-10')
vstack([a,b,c],24).save(W+'pnfig/pn_benchmark_curves.png')
g=[Image.open(A+'ceea9644-image.png').convert('RGB'), o('pn/f84ef9cc/c011_0.png'), o('pn/5bdc9ced/c016_0.png')]
t=['(a) Supplied example: activation within the right upper and middle lung zone',
   '(b) DenseNet121-20: normal radiograph (validation) misclassified as pneumonia, p = 0.86',
   '(c) VGG16-20: pneumonia radiograph (test) misclassified as normal, p(pneumonia) = 0.12']
vstack([lab(i,s) for i,s in zip(g,t)],16).save(W+'pnfig/pn_gradcam.png')
for f in ['pn_hybrid_eval','pn_benchmark_curves','pn_gradcam']: print(f, Image.open(W+'pnfig/'+f+'.png').size)
