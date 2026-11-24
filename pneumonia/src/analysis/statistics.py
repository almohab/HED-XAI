"""Run from the pneumonia/ folder (analysis of the recorded results; paths may need adjusting)."""
"""Pneumonia metrics recomputed from confusion matrices ([[TN, FP], [FN, TP]]; positive = Pneumonia)."""
import json, math
from scipy.stats import norm

# Re-split Kermany data (test n = 880: 238 Normal / 642 Pneumonia)
M = {
 "HED-XAI-PN (proposed)": dict(cm=[[227, 11], [3, 639]], ep="50 (best @ 6)", auc=0.9991, ap=0.9987, params=2728226, size=26.10, ms=14.84,
                               train_min=86.96, cw="no", src="Hybird_Model_Pneumonia_PhD.ipynb"),
 "DenseNet121-20": dict(cm=[[234, 4], [28, 614]], ep="20", auc=0.9966, ap=0.9987, params=7300161, size=None, ms=88.36, cw="yes",
                        src="Final_models_Pneumonia (cell 19)"),
 "InceptionV3-10": dict(cm=[[231, 7], [25, 617]], ep="10", auc=0.9949, ap=0.9981, params=22360353, size=129.51, ms=126.40, train_min=119.33,
                        cw="no", src="Final_models_Pneumonia (cell 28)"),
 "MobileNetV2-20": dict(cm=[[234, 4], [53, 589]], ep="20", auc=0.9952, ap=0.9982, params=2586177, size=None, ms=50.44, cw="yes",
                        src="Final_models_Pneumonia (cell 25)"),
 "VGG16-20": dict(cm=[[236, 2], [85, 557]], ep="20", auc=0.9964, ap=0.9986, params=21137729, size=None, ms=274.88, cw="yes",
                  src="Final_models_Pneumonia (cell 5)"),
 "VGG16-10": dict(cm=[[236, 2], [93, 549]], ep="10", auc=0.995, ap=0.998, params=21137729, size=None, ms=290.97, cw="yes",
                  src="Final_models_Pneumonia (cell 4)"),
 "CNN-A-20": dict(cm=[[224, 14], [155, 487]], ep="20", auc=0.9448, ap=None, params=13235905, size=151.55, ms=15.53, train_min=62.23, cw="yes",
                  src="earlier run; the supplied notebook copy was interrupted at epoch 15", untraceable=True),
}
# Original Kermany partition (test n = 624: 234 Normal / 390 Pneumonia), exploratory
ORIG = {"VGG16-10 (original split)": dict(cm=[[162, 72], [3, 387]], ep="10", auc=0.9671, cw="no", src="Final_models_Pneumonia (cell 0)")}

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

out = {"main": {}, "orig": {}, "cmp": {}}
for grp, D in (("main", M), ("orig", ORIG)):
    for k, d in D.items():
        m = metrics(d["cm"]); out[grp][k] = dict(d, **m)
        print(f"{k:26s} acc {m['acc']*100:6.2f} [{m['lo']*100:.2f}-{m['hi']*100:.2f}] sens {m['sens']*100:6.2f} [{m['sens_lo']*100:.2f}-{m['sens_hi']*100:.2f}] "
              f"spec {m['spec']*100:6.2f} [{m['spec_lo']*100:.2f}-{m['spec_hi']*100:.2f}] prec {m['prec']*100:6.2f} npv {m['npv']*100:6.2f} "
              f"F1 {m['f1']:.4f} mF1 {m['mf1']:.4f} bAcc {m['bacc']*100:.2f} MCC {m['mcc']:.4f} k {m['kappa']:.4f} err {m['err']}")
h = out["main"]["HED-XAI-PN (proposed)"]
for k, o in out["main"].items():
    if k.startswith("HED"): continue
    n = 880; p = (h["acc"] + o["acc"]) / 2; z = (h["acc"] - o["acc"]) / math.sqrt(2 * p * (1 - p) / n)
    out["cmp"][k] = dict(diff=(h["acc"] - o["acc"]) * 100, z=z, p=2 * (1 - norm.cdf(abs(z))))
    # specificity comparison (238 normals)
    ps = (h["spec"] + o["spec"]) / 2; zs = (h["spec"] - o["spec"]) / math.sqrt(2 * ps * (1 - ps) / 238)
    out["cmp"][k].update(spec_diff=(h["spec"] - o["spec"]) * 100, z_spec=zs, p_spec=2 * (1 - norm.cdf(abs(zs))))
    print(f"HED vs {k:16s} acc diff {out['cmp'][k]['diff']:+.2f} z {z:.2f} p {out['cmp'][k]['p']:.4f} | spec diff {out['cmp'][k]['spec_diff']:+.2f} p {out['cmp'][k]['p_spec']:.3f}")
json.dump(out, open("results/metrics_all_models.json", "w"), indent=1)
