"""Run from the skin_cancer/ folder (analysis of the recorded results; paths may need adjusting)."""
"""Skin cancer metrics recomputed from confusion matrices ([[TN, FP], [FN, TP]]; positive = Malignant). Test n = 660 (360 benign, 300 malignant)."""
import json, math
from scipy.stats import norm

M = {
 "HED-XAI-SC (proposed)": dict(cm=[[315, 45], [35, 265]], input=224, ep="10 / 20 (best @ 10)", auc=0.9595, ap=0.9547, params=2728097, size=26.10,
                               ms=18.14, ms_runs=[18.14, 20.83], train_min=14.95, train_min_20=31.52, src="Final_HED-XAI-SC_Skin_Cancer_.ipynb"),
 "DenseNet121-50 (ES@10)": dict(cm=[[300, 60], [28, 272]], input=224, ep="50 (stopped @ 10, best @ 4)", auc=0.946, params=7300161, size=36.16, src="cell 10"),
 "DenseNet121-10": dict(cm=[[307, 53], [36, 264]], input=224, ep="10", auc=0.952, params=7300161, size=36.16, src="cell 5"),
 "DenseNet121-40 (ES@13)": dict(cm=[[304, 56], [33, 267]], input=224, ep="40 (stopped @ 13, best @ 7)", auc=0.949, params=7300161, size=36.16, src="cell 9"),
 "VGG16-GAP-20 (ES@12)": dict(cm=[[297, 63], [29, 271]], input=224, ep="20 (stopped @ 12, best @ 7)", auc=0.944, params=14846273, size=111.74, src="cell 7"),
 "VGG16-Flatten-50 (ES@24)": dict(cm=[[289, 71], [33, 267]], input=150, ep="50 (stopped @ 24, best @ 19)", auc=None, params=18910017, src="cell 2"),
 "MobileNetV2-50": dict(cm=[[308, 52], [53, 247]], input=150, ep="50", auc=0.92, params=2586177, acc_eval=82.73, src="cell 12"),
 "VGG16-Flatten-10": dict(cm=[[278, 82], [31, 269]], input=150, ep="10", auc=None, params=18910017, src="cell 3"),
 "InceptionV3-10": dict(cm=[[339, 21], [99, 201]], input=299, ep="10", auc=0.9217, params=22360353, size=129.51, ms=136.93, train_min=70.08, src="cell 16"),
 "Xception-10": dict(cm=[[309, 51], [71, 229]], input=224, ep="10 (best @ 10)", auc=0.90, params=21123881, src="cell 14"),
 "CNN-20": dict(cm=[[248, 112], [15, 285]], input=150, ep="20", auc=None, params=21309569, src="cell 1"),
 "CNN-10": dict(cm=[[272, 88], [40, 260]], input=150, ep="10", auc=None, params=21309569, src="cell 0"),
 "Xception-20": dict(cm=[[309, 51], [85, 215]], input=224, ep="20 (best @ 17)", auc=0.90, params=21123881, src="cell 15"),
}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def metrics(cm):
    (tn, fp), (fn, tp) = cm; n = tn + fp + fn + tp
    acc = (tp + tn) / n; sens = tp / (tp + fn); spec = tn / (tn + fp); prec = tp / (tp + fp); npv = tn / (tn + fn)
    f1 = 2 * prec * sens / (prec + sens); f1n = 2 * npv * spec / (npv + spec)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2; kappa = (acc - pe) / (1 - pe)
    lo, hi = wilson(tp + tn, n); slo, shi = wilson(tp, tp + fn); plo, phi = wilson(tn, tn + fp)
    return dict(n=n, tn=tn, fp=fp, fn=fn, tp=tp, acc=acc, lo=lo, hi=hi, sens=sens, sens_lo=slo, sens_hi=shi, spec=spec, spec_lo=plo,
                spec_hi=phi, prec=prec, npv=npv, f1=f1, mf1=(f1 + f1n) / 2, bacc=(sens + spec) / 2, mcc=mcc, kappa=kappa, err=fp + fn)

out = {"main": {}, "cmp": {}}
for k, d in M.items():
    m = metrics(d["cm"]); out["main"][k] = dict(d, **m)
    print(f"{k:26s} acc {m['acc']*100:6.2f} [{m['lo']*100:.2f}-{m['hi']*100:.2f}] sens {m['sens']*100:6.2f} spec {m['spec']*100:6.2f} prec {m['prec']*100:6.2f} "
          f"npv {m['npv']*100:6.2f} F1 {m['f1']:.4f} bAcc {m['bacc']*100:.2f} MCC {m['mcc']:.4f} err {m['err']} FN {m['fn']} FP {m['fp']}")
h = out["main"]["HED-XAI-SC (proposed)"]
for k, o in out["main"].items():
    if k.startswith("HED"): continue
    n = 660; p = (h["acc"] + o["acc"]) / 2; z = (h["acc"] - o["acc"]) / math.sqrt(2 * p * (1 - p) / n)
    out["cmp"][k] = dict(diff=(h["acc"] - o["acc"]) * 100, z=z, p=2 * (1 - norm.cdf(abs(z))))
    print(f"HED vs {k:26s} diff {out['cmp'][k]['diff']:+.2f} p {out['cmp'][k]['p']:.4f}")
json.dump(out, open("results/metrics_all_models.json", "w"), indent=1)
