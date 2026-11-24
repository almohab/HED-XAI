"""Run from the skin_cancer/ folder (analysis of the recorded results; paths may need adjusting)."""
import json, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a',
                     'axes.labelcolor': '#333', 'xtick.color': '#444', 'ytick.color': '#333'})
W = 'results/'; OUT = W + 'scfig/'
S = json.load(open(W + 'sc_stats.json')); SM, SO = S['main'], None
HYB = "HED-XAI-SC (proposed)"
C1, C2, C3, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8c8c8c'
cmap = LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#9cc3ee', '#2a78d6', '#0d3f7a'])
lab = ['Benign', 'Malig.']

def cm_ax(ax, k, d, hyb=False, extra=""):
    m = np.array(d['cm']); rown = m / m.sum(1, keepdims=True)
    ax.imshow(rown, cmap=cmap, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            v = m[i, j]; col = 'white' if rown[i, j] > 0.55 else '#1a1a1a'
            ax.text(j, i - 0.08, str(v), ha='center', va='center', fontsize=13, weight='bold', color=col)
            ax.text(j, i + 0.22, f"{rown[i, j]*100:.1f}%", ha='center', va='center', fontsize=7.6, color=col)
    ax.set_xticks(range(2)); ax.set_yticks(range(2)); ax.set_xticklabels(lab, fontsize=9); ax.set_yticklabels(lab, fontsize=9)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"{k}{extra}\nAcc {d['acc']*100:.2f}% · FN {d['fn']} · FP {d['fp']}", fontsize=9.2,
                 weight='bold' if hyb else 'normal', color='#1F3864' if hyb else '#222')
    if hyb:
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(C2); s.set_linewidth(2.5)

# ---------------- confusion matrices: proposed + 7 representative benchmarks
order = [HYB, "DenseNet121-50 (ES@10)", "DenseNet121-10", "VGG16-GAP-20 (ES@12)", "VGG16-Flatten-50 (ES@24)", "MobileNetV2-50", "InceptionV3-10", "CNN-20"]
fig, axs = plt.subplots(2, 4, figsize=(13, 7.0), dpi=220)
for ax, k in zip(axs.flat, order):
    cm_ax(ax, k.replace(" (proposed)", ""), SM[k], k == HYB)
for ax in axs[:, 0]: ax.set_ylabel('Actual class', fontsize=9)
for ax in axs[1, :]: ax.set_xlabel('Predicted class', fontsize=9)
fig.text(0.5, 0.008, 'Test set: benign n = 360, malignant n = 300. Percentages are row-normalised. ES@k = EarlyStopping at epoch k with restoration of the best weights.',
         ha='center', fontsize=8.2, color='#555')
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.subplots_adjust(hspace=0.45); plt.savefig(OUT + 'sc_confusion_panel.png', facecolor='white'); plt.close()

# ---------------- operating points + accuracy/balanced accuracy
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 6.0), dpi=220, gridspec_kw={'width_ratios': [1, 1.2]})
OFF = {HYB: (8, 2), "DenseNet121-50 (ES@10)": (8, 4), "DenseNet121-10": (-8, 6), "DenseNet121-40 (ES@13)": (8, -9), "VGG16-GAP-20 (ES@12)": (8, -2), "VGG16-Flatten-50 (ES@24)": (8, -4), "MobileNetV2-50": (-8, -10), "VGG16-Flatten-10": (8, -2), "InceptionV3-10": (8, 0), "Xception-10": (8, 3), "CNN-20": (8, 0), "CNN-10": (8, -3), "Xception-20": (8, -10)}
POS = {HYB: (3.5, 84.0), "DenseNet121-50 (ES@10)": (6.5, 94.5), "DenseNet121-10": (5.0, 86.0), "DenseNet121-40 (ES@13)": (4.0, 91.5),
       "VGG16-GAP-20 (ES@12)": (18.5, 94.0), "VGG16-Flatten-50 (ES@24)": (24.5, 86.5), "VGG16-Flatten-10": (25.0, 91.5),
       "MobileNetV2-50": (3.0, 80.0), "InceptionV3-10": (8.0, 66.5), "Xception-10": (17.0, 77.0), "Xception-20": (17.0, 70.5), "CNN-20": (27.0, 97.5), "CNN-10": (27.5, 84.0)}
for k, d in SM.items():
    c = C2 if k == HYB else C1
    x, y = (1 - d['spec']) * 100, d['sens'] * 100
    a1.errorbar(x, y, xerr=[[x - (1 - d['spec_hi']) * 100], [(1 - d['spec_lo']) * 100 - x]], yerr=[[y - d['sens_lo'] * 100], [d['sens_hi'] * 100 - y]],
                fmt='none', ecolor='#cfcfcf', elinewidth=0.8, zorder=1)
    a1.scatter(x, y, s=80 if k == HYB else 44, color=c, edgecolor='white', lw=1.3, zorder=3)
    tx, ty = POS[k]
    a1.annotate(k.replace(" (proposed)", ""), (x, y), xytext=(tx, ty), textcoords='data', fontsize=7.8, ha='left', va='center',
                color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal',
                arrowprops=dict(arrowstyle='-', color='#9a9a9a', lw=0.6, shrinkA=1, shrinkB=3))
a1.set_xlim(0, 36); a1.set_ylim(60, 100); a1.set_xlabel('False-positive rate = 100 − specificity (%)', fontsize=9.5); a1.set_ylabel('Sensitivity (%)', fontsize=9.5)
a1.grid(True, color='#ececec'); a1.set_axisbelow(True); a1.set_title('(a) Operating points (Wilson 95% CIs)', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
models = sorted(SM, key=lambda k: SM[k]['acc']); y = np.arange(len(models))
for yi, k in zip(y, models):
    d = SM[k]; c = C2 if k == HYB else C1
    a2.plot([d['lo'] * 100, d['hi'] * 100], [yi + 0.14] * 2, color=c, lw=1.6, solid_capstyle='round')
    a2.scatter(d['acc'] * 100, yi + 0.14, s=60 if k == HYB else 40, color=c, edgecolor='white', lw=1.2, zorder=3, label='Accuracy' if yi == 0 else None)
    a2.scatter(d['bacc'] * 100, yi - 0.16, s=46 if k == HYB else 34, marker='D', color='white', edgecolor=c, lw=1.6, zorder=3, label='Balanced accuracy' if yi == 0 else None)
    a2.text(92.3, yi, f"{d['acc']*100:.2f} / {d['bacc']*100:.2f}", va='center', fontsize=8.2, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a2.set_yticks(y); a2.set_yticklabels([k.replace(" (proposed)", "") for k in models], fontsize=9)
for lbl, k in zip(a2.get_yticklabels(), models):
    if k == HYB: lbl.set_weight('bold'); lbl.set_color('#1F3864')
a2.set_xlim(72, 97); a2.xaxis.set_major_locator(FixedLocator([75, 80, 85, 90])); a2.set_xlabel('Percentage (accuracy with Wilson 95% CI)', fontsize=9.5)
a2.xaxis.grid(True, color='#e6e6e6'); a2.set_axisbelow(True); a2.legend(frameon=False, fontsize=8.6, loc='upper left')
a2.set_title('(b) Accuracy and balanced accuracy', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
plt.tight_layout(); plt.savefig(OUT + 'sc_operating_points.png', facecolor='white'); plt.close()

# ---------------- HED-XAI-SC training dynamics (20-epoch run, parsed from the notebook log)
import os
A = "/mnt/user-data/attachments/00294aa0-4669-5e77-944b-0b0344daf709/"
f = [x for x in os.listdir(A) if x.startswith("177b60d9")][0]; nb = json.load(open(A + f))
t = "".join("".join(o.get("text", "")) for o in nb["cells"][1]["outputs"] if o.get("output_type") == "stream")
t = re.sub(r"\x1b\[[0-9;]*m", "", t)
rows = re.findall(r"Epoch (\d+)/20\n.*?accuracy: ([\d.]+) - auc: [\d.]+ - loss: ([\d.]+).*?val_accuracy: ([\d.]+) - val_auc: [\d.]+ - val_loss: ([\d.]+).*?learning_rate: ([\d.e+-]+)", t, flags=re.S)
ep = np.array(rows, dtype=float); e = ep[:, 0]
json.dump({'epoch': e.tolist(), 'acc': ep[:, 1].tolist(), 'loss': ep[:, 2].tolist(), 'val_acc': ep[:, 3].tolist(), 'val_loss': ep[:, 4].tolist(), 'lr': ep[:, 5].tolist()},
          open(OUT + 'hedxai_sc_history_20ep.json', 'w'), indent=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), dpi=220)
for ax, tr, va, yl, ttl in [(a1, ep[:, 1] * 100, ep[:, 3] * 100, 'Accuracy (%)', '(a) Accuracy'), (a2, ep[:, 2], ep[:, 4], 'Binary cross-entropy loss', '(b) Loss')]:
    ax.plot(e, tr, color=C1, lw=2, marker='o', ms=4, label='Training')
    ax.plot(e, va, color=C2, lw=2, marker='s', ms=4, label='Validation')
    ax.axvline(10, color='#555', ls='--', lw=1); ax.set_xticks(range(1, 21)); ax.set_xlabel('Epoch', fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5)
    ax.axvspan(10.5, 20.5, color='#f2f2f2', zorder=0)
    ax.grid(True, color='#ededed'); ax.set_axisbelow(True); ax.set_title(ttl, loc='left', fontsize=10.5, weight='bold', color='#1F3864'); ax.set_xlim(0.5, 20.5)
a1.set_ylim(64, 100); a1.legend(frameon=False, fontsize=9, loc='lower right')
a1.annotate('best val_loss (epoch 10)\n→ checkpoint', xy=(10, 85.6), xytext=(4.2, 70), fontsize=8.3, color='#333', arrowprops=dict(arrowstyle='-', color='#777', lw=0.8))
a1.text(15.5, 74, 'epochs 11–20 of the 20-epoch run:\nno further improvement', ha='center', fontsize=8.2, color='#666')
plt.tight_layout(); plt.savefig(OUT + 'sc_training_curves.png', facecolor='white'); plt.close()

# ---------------- efficiency trade-off
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=220)
OFF = {HYB: (8, 3), "DenseNet121-50 (ES@10)": (8, 5), "DenseNet121-10": None, "DenseNet121-40 (ES@13)": None, "VGG16-GAP-20 (ES@12)": (8, -3), "VGG16-Flatten-50 (ES@24)": (8, 3),
       "MobileNetV2-50": (8, -3), "VGG16-Flatten-10": (8, -9), "InceptionV3-10": (8, 3), "Xception-10": (-8, 4), "CNN-20": (8, 3), "CNN-10": (8, -9), "Xception-20": (-8, -9)}
for k, d in SM.items():
    c = C2 if k == HYB else C1
    a1.scatter(d['params'] / 1e6, d['acc'] * 100, s=80 if k == HYB else 44, color=c, edgecolor='white', lw=1.3, zorder=3)
    o = OFF[k]
    if o is None: continue
    lbl = "DenseNet121 (3 runs: 10, 40, 50)" if k.startswith("DenseNet121-50") else k.replace(" (proposed)", "")
    a1.annotate(lbl, (d['params'] / 1e6, d['acc'] * 100), xytext=o, textcoords='offset points',
                           ha='right' if o[0] < 0 else 'left', fontsize=7.6, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a1.set_xscale('log'); a1.set_xlim(2, 40); a1.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 40])); a1.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '20', '40'])); a1.xaxis.set_minor_locator(NullLocator())
a1.set_ylim(78.5, 89); a1.set_xlabel('Parameters (millions, log scale)', fontsize=9.5); a1.set_ylabel('Test accuracy (%)', fontsize=9.5)
a1.grid(True, color='#ededed'); a1.set_axisbelow(True); a1.set_title('(a) Accuracy versus model size', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
names = [HYB, "DenseNet121-10", "VGG16-GAP-20 (ES@12)", "InceptionV3-10"]; yy = np.arange(len(names))[::-1]
a2.barh(yy, [SM[k]['size'] for k in names], 0.55, color=[C2 if k == HYB else C1 for k in names])
for yi, k in zip(yy, names):
    extra = f"  ·  {SM[k]['ms']:.2f} ms/img" if SM[k].get('ms') else ""
    a2.text(SM[k]['size'] + 2, yi, f"{SM[k]['size']:.2f} MB{extra}", va='center', fontsize=8.5, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a2.set_yticks(yy); a2.set_yticklabels([k.replace(" (proposed)", "") for k in names], fontsize=9); a2.set_xlim(0, 200)
a2.set_xlabel('Saved model size (MB)', fontsize=9.5); a2.xaxis.grid(True, color='#ededed'); a2.set_axisbelow(True)
a2.set_title('(b) Saved model size (and inference time where recorded)', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
plt.tight_layout(); plt.savefig(OUT + 'sc_tradeoff.png', facecolor='white'); plt.close()
print('ok', len(rows))
