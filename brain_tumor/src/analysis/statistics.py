"""Run from the brain_tumor/ folder."""
import numpy as np, json, math
from scipy.stats import norm
CLS=["Glioma","Meningioma","Pituitary"]
M={
 "HED-XAI-BT (proposed)": dict(cm=[[194,18,3],[4,98,5],[0,4,136]], ep="50 (best @ 11)", auc=0.9910, size=26.10, ms=13.70, params=2728355, train_min=32.68, src="Hybrid_model_Brain_Tumor__PhD.ipynb"),
 "CNN-100": dict(cm=[[208,7,0],[12,95,0],[3,2,135]], ep="100", size=254.66, src="Brain_Tumor_Custom_CNN_100_Epochs.ipynb"),
 "VGG16-20": dict(cm=[[197,17,1],[2,105,0],[0,7,133]], ep="20 (best @ 16)", auc=0.9914, size=203.25, ms=260.44, params=27561795, train_min=13308.04/60, src="Brain_Tumor_VGG16_20_Epochs.ipynb (re-run with checkpoint)"),
 "CNN-50": dict(cm=[[195,19,1],[7,100,0],[2,2,136]], ep="50", size=254.66, src="CNN_model_50_Epochs_brain_tumer.ipynb"),
 "VGG16-10": dict(cm=[[195,17,3],[4,101,2],[0,6,134]], ep="10", size=203.25, src="VGG16_model_10_Epochs_Brain_Tumer.ipynb"),
 "DenseNet201-20": dict(cm=[[192,22,1],[9,94,4],[2,5,133]], ep="20", size=83.25, src="Brain_Tumor_DenseNet201_20_Epochs.ipynb"),
 "MobileNetV2-10": dict(cm=[[191,24,0],[10,92,5],[0,7,133]], ep="10", size=14.14, src="MobileneteV2_Brain_Tumer_10_epochs.ipynb"),
 "MobileNetV2-20": dict(cm=[[189,22,4],[9,90,8],[0,4,136]], ep="20", size=14.14, src="Brain_Tumor_MobileNetV2_20_Epochs.ipynb"),
 "DenseNet201-10": dict(cm=[[194,18,3],[15,87,5],[4,5,131]], ep="10", size=83.25, src="DenseNet201_model_10_Epochs_Brain_Tumer.ipynb"),
 "InceptionV3-20": dict(cm=[[194,16,5],[14,83,10],[0,5,135]], ep="20", size=96.14, src="Brain_Tumor_InceptionV3_20_Epochs.ipynb"),
 "InceptionV3-10": dict(cm=[[189,21,5],[7,91,9],[1,12,127]], ep="10", size=96.14, src="InceptionV3_model_10_Epochs_Brain_Tumer_.ipynb"),
}
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c-h,c+h
out={}
for k,d in M.items():
    cm=np.array(d['cm']); n=cm.sum(); tp=np.diag(cm); fn=cm.sum(1)-tp; fp=cm.sum(0)-tp; tn=n-tp-fn-fp
    rec=tp/(tp+fn); prec=tp/(tp+fp); spec=tn/(tn+fp); f1=2*prec*rec/(prec+rec)
    acc=tp.sum()/n; lo,hi=wilson(tp.sum(),n)
    # multiclass MCC (Gorodkin) and kappa
    t=cm.sum(1); p=cm.sum(0); c=tp.sum()
    mcc=(c*n-(t*p).sum())/math.sqrt((n*n-(p*p).sum())*(n*n-(t*t).sum()))
    pe=(t*p).sum()/n**2; kappa=(acc-pe)/(1-pe)
    out[k]=dict(d, acc=acc,lo=lo,hi=hi,rec=rec.tolist(),prec=prec.tolist(),spec=spec.tolist(),f1=f1.tolist(),
        mprec=prec.mean(),mrec=rec.mean(),mf1=f1.mean(),mspec=spec.mean(),mcc=mcc,kappa=kappa,err=int(n-c),n=int(n))
    print(f"{k:24s} acc {acc*100:6.2f} [{lo*100:.2f},{hi*100:.2f}] mP {prec.mean():.4f} mR {rec.mean():.4f} mF1 {f1.mean():.4f} mSpec {spec.mean()*100:.2f} MCC {mcc:.4f} k {kappa:.4f} err {n-c} | rec "+" ".join(f"{r*100:.1f}" for r in rec)+" | spec "+" ".join(f"{s*100:.2f}" for s in spec))
h=out["HED-XAI-BT (proposed)"]
for k in ["CNN-100","VGG16-20","CNN-50","DenseNet201-20","MobileNetV2-20"]:
    o=out[k]; p=(h['acc']+o['acc'])/2; z=(h['acc']-o['acc'])/math.sqrt(2*p*(1-p)/462); print(k, f"diff {100*(h['acc']-o['acc']):.2f} z {z:.2f} p {2*(1-norm.cdf(abs(z))):.3f}")
json.dump(out,open('results/metrics_all_models.json','w'),indent=1)
