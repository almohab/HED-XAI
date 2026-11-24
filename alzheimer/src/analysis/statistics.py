"""Run from the alzheimer/ folder."""
import math, json
from scipy.stats import norm
# (TN, FP, FN, TP) — positive = Very mild Dementia; all n=600 (300/300)
M = {
 "HED-XAI-AD (VGG16–InceptionV3 + CA)": dict(cm=(276,24,19,281), auc=0.9843, ap=0.9828, ep="10+40", size=146.37, ms=97.56, params=38070433, train_min=480.89),
 "HED-XAI-AD-V2 (VGG16–DenseNet121 + CBAM)": dict(cm=(234,66,20,280), auc=0.9094, ap=0.8769, ep="20", size=223.80, ms=237.03, params=23855748, train_min=250.01),
 "VGG16-50": dict(cm=(263,37,21,279), auc=0.97, ap=0.96, ep="50", size=184.11, ms=272.66, params=21170499),
 "InceptionV3-50": dict(cm=(277,23,37,263), auc=0.97, ap=0.97, ep="50", size=129.51, ms=112.44),
 "VGG16-20": dict(cm=(254,46,28,272), auc=0.95, ap=0.93, ep="20", size=184.11, ms=278.29, params=21170499),
 "VGG16-10": dict(cm=(222,78,12,288), auc=0.93, ap=None, ep="10", size=None, ms=None, params=21170499),
 "MobileNetV2 (B)-50": dict(cm=(258,42,53,247), auc=0.93, ap=0.92, ep="50 (ES)", size=22.78, ms=23.32),
 "CNN-50": dict(cm=(201,99,2,298), auc=0.93, ap=0.92, ep="50", size=112.93, ms=15.38),
 "MobileNetV2 (A)-30": dict(cm=(196,104,10,290), auc=0.90, ap=0.89, ep="20+10", size=22.78, ms=25.41),
 "CNN-20": dict(cm=(229,71,44,256), auc=0.89, ap=0.88, ep="20", size=112.94, ms=38.69),
}
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
out={}
for name,d in M.items():
    TN,FP,FN,TP=d['cm']; n=TN+FP+FN+TP
    acc=(TP+TN)/n; sens=TP/(TP+FN); spec=TN/(TN+FP); prec=TP/(TP+FP); npv=TN/(TN+FN)
    f1=2*prec*sens/(prec+sens)
    f1n=2*npv*spec/(npv+spec); macro=(f1+f1n)/2
    mcc=(TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    po=acc; pe=((TP+FP)*(TP+FN)+(TN+FN)*(TN+FP))/n**2; kappa=(po-pe)/(1-pe)
    lo,hi=wilson(TP+TN,n)
    out[name]=dict(acc=acc,lo=lo,hi=hi,sens=sens,spec=spec,prec=prec,npv=npv,f1=f1,macro=macro,mcc=mcc,kappa=kappa,bacc=(sens+spec)/2, **d)
    print(f"{name:45s} acc {acc*100:6.2f} [{lo*100:5.2f},{hi*100:5.2f}] sens {sens*100:6.2f} spec {spec*100:6.2f} prec {prec*100:6.2f} npv {npv*100:6.2f} F1 {f1:.4f} mF1 {macro:.4f} MCC {mcc:.4f} k {kappa:.4f} err {FP+FN}")
# two-proportion z test hybrid vs others (unpaired, approximate)
h=out["HED-XAI-AD (VGG16–InceptionV3 + CA)"]
for k in ["VGG16-50","InceptionV3-50","HED-XAI-AD-V2 (VGG16–DenseNet121 + CBAM)","VGG16-20"]:
    o=out[k]; n=600
    p1,p2=h['acc'],o['acc']; p=(p1+p2)/2; z=(p1-p2)/math.sqrt(2*p*(1-p)/n); pv=2*(1-norm.cdf(abs(z)))
    print(f"vs {k}: diff {100*(p1-p2):.2f} pp, z={z:.2f}, p={pv:.4f}; spec diff z:", end=' ')
    s1,s2=h['spec'],o['spec']; p=(s1+s2)/2; z2=(s1-s2)/math.sqrt(2*p*(1-p)/300); print(f"{100*(s1-s2):.2f}pp z={z2:.2f} p={2*(1-norm.cdf(abs(z2))):.4f}")
json.dump(out,open('results/metrics_all_models.json','w'),indent=1,default=str)
