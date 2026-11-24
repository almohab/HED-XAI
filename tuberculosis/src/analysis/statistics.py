"""Run from the tuberculosis/ folder (analysis of the recorded results; paths may need adjusting)."""
"""Tuberculosis metrics recomputed from confusion matrices ([[TN, FP], [FN, TP]]; positive = Tuberculosis)."""
import json, math
from scipy.stats import norm, fisher_exact

# Protocol B: de-duplicated collection (6,670 images), test n = 1,001 (525 Normal / 476 TB)
B = {
 "HED-XAI-TB (proposed)": dict(cm=[[525, 0], [5, 471]], ep="10 / 20 (best @ 6)", auc=0.999996, params=2728097, size=25.89,
                               ms=10.86, ms_runs=[10.86, 15.04, 20.65], train_min=18.31, train_min_20=35.25,
                               src="Hybrid_model_TB_PhD.ipynb", derived=False),
 "VGG16 (clean)": dict(cm=[[525, 0], [3, 473]], ep="—", auc=1.0000, params=14879041, size=112.13, ms=86.92, train_min=162.76,
                       src="not among the supplied notebooks", derived=True),
 "DenseNet201-20": dict(cm=[[525, 0], [5, 471]], ep="20", auc=0.9998, params=18831297, size=None, ms=74.92, train_min=130.52,
                        src="not among the supplied notebooks", derived=True),
 "InceptionV3-20": dict(cm=[[525, 0], [6, 470]], ep="20", auc=0.9993, params=22344865, size=None, ms=None, train_min=None,
                        src="not among the supplied notebooks", derived=True),
}
# Protocol A: original 7,000 images (330 duplicates retained), test n = 1,050 (525 / 525)
# cm from model.predict on the test generator; acc_eval = value printed by model.evaluate()
A = {
 "CNN-20":         dict(cm=[[517, 8], [14, 511]], ep=20, params=14821121, acc_eval=97.90, src="custom_cnn_20epochs_Tuberculosis.ipynb", input="128×128 grey"),
 "CNN-50":         dict(cm=[[511, 14], [2, 523]], ep=50, params=14821121, acc_eval=98.48, src="custom_cnn_50epochs_Tuberculosis.ipynb", input="128×128 grey"),
 "CNN-100":        dict(cm=[[525, 0], [15, 510]], ep=100, params=14821121, acc_eval=98.57, src="custom_cnn_100epochs_Tuberculosis.ipynb", input="128×128 grey"),
 "VGG16-10":       dict(cm=[[509, 16], [4, 521]], ep=10, params=27691843, acc_eval=97.14, src="vgg16_densenet201_10epochs_Tuberculosis.ipynb", input="224×224 RGB"),
 "VGG16-20":       dict(cm=[[516, 9], [1, 524]], ep=20, params=27691843, acc_eval=99.52, src="vgg16_20epochs_Tuberculosis_old.ipynb", input="224×224 RGB"),
 "DenseNet201-10": dict(cm=[[524, 1], [12, 513]], ep=10, params=19437123, acc_eval=98.95, auc=0.9994, src="vgg16_densenet201_10epochs_Tuberculosis.ipynb", input="224×224 RGB"),
 "InceptionV3-10": dict(cm=[[518, 7], [13, 512]], ep=10, params=22983459, acc_eval=97.90, src="inceptionv3_10_20epochs_Tuberculosis.ipynb", input="299×299 RGB"),
 "InceptionV3-20": dict(cm=[[520, 5], [8, 517]], ep=20, params=22983459, acc_eval=98.86, src="inceptionv3_10_20epochs_Tuberculosis.ipynb", input="299×299 RGB"),
 "MobileNetV2-10": dict(cm=[[519, 6], [8, 517]], ep=10, params=3045443, acc_eval=98.86, src="mobilenetv2_10epochs_Tuberculosis.ipynb", input="224×224 RGB"),
}

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

def metrics(cm):
    (tn, fp), (fn, tp) = cm; n = tn + fp + fn + tp
    acc = (tp + tn) / n; sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else float("nan"); npv = tn / (tn + fn)
    f1 = 2 * prec * sens / (prec + sens)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2; kappa = (acc - pe) / (1 - pe)
    lo, hi = wilson(tp + tn, n); slo, shi = wilson(tp, tp + fn); plo, phi = wilson(tn, tn + fp)
    return dict(n=n, tn=tn, fp=fp, fn=fn, tp=tp, acc=acc, lo=lo, hi=hi, sens=sens, sens_lo=slo, sens_hi=shi,
                spec=spec, spec_lo=plo, spec_hi=phi, prec=prec, npv=npv, f1=f1, mcc=mcc, kappa=kappa, err=fp + fn)

out = {"B": {}, "A": {}}
for proto, M in (("B", B), ("A", A)):
    print(f"\n=== Protocol {proto}")
    for k, d in M.items():
        m = metrics(d["cm"]); out[proto][k] = dict(d, **m)
        print(f"{k:24s} acc {m['acc']*100:6.2f} [{m['lo']*100:.2f}-{m['hi']*100:.2f}] sens {m['sens']*100:6.2f} [{m['sens_lo']*100:.2f}-{m['sens_hi']*100:.2f}] "
              f"spec {m['spec']*100:6.2f} [{m['spec_lo']*100:.2f}-{m['spec_hi']*100:.2f}] prec {m['prec']*100:6.2f} npv {m['npv']*100:6.2f} "
              f"F1 {m['f1']:.4f} MCC {m['mcc']:.4f} k {m['kappa']:.4f} FN {m['fn']} FP {m['fp']}" + (f" | evaluate() {d['acc_eval']:.2f}" if 'acc_eval' in d else ""))

# Pairwise comparison against HED-XAI-TB (Protocol B): FP = 0 for all, so models differ only in FN among 476 TB cases.
h = out["B"]["HED-XAI-TB (proposed)"]
cmp = {}
for k, o in out["B"].items():
    if k.startswith("HED"): continue
    p = (h["acc"] + o["acc"]) / 2; z = (h["acc"] - o["acc"]) / math.sqrt(2 * p * (1 - p) / 1001)
    pz = 2 * (1 - norm.cdf(abs(z)))
    _, pf = fisher_exact([[h["fn"], h["tp"]], [o["fn"], o["tp"]]])
    cmp[k] = dict(diff=(h["acc"] - o["acc"]) * 100, z=z, p_z=pz, p_fisher_sens=pf)
    print(f"HED vs {k:16s} diff {cmp[k]['diff']:+.2f} pp  z {z:+.2f} p {pz:.3f}  Fisher(sens) p {pf:.3f}")
out["cmp"] = cmp

# Expected number of Protocol A test images with an exact copy in the training partition
# (330 removed duplicates, assumed to form 330 pairs within the TB class; random 70/15/15 split)
exp_pairs = 330 * 2 * 0.15 * 0.70
out["leak_expected"] = exp_pairs
print(f"\nExpected Protocol A test images with a byte-identical twin in training: ~{exp_pairs:.0f} (of 525 TB test images)")
json.dump(out, open("results/metrics_all_models.json", "w"), indent=1)
