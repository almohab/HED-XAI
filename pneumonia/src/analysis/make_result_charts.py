"""Run from the pneumonia/ folder (analysis of the recorded results; paths may need adjusting)."""
import json, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a',
                     'axes.labelcolor': '#333', 'xtick.color': '#444', 'ytick.color': '#333'})
W = 'results/'; OUT = W + 'pnfig/'
S = json.load(open(W + 'pn_stats.json')); SM, SO = S['main'], S['orig']
HYB = "HED-XAI-PN (proposed)"
C1, C2, C3, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8c8c8c'
cmap = LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#9cc3ee', '#2a78d6', '#0d3f7a'])
lab = ['Normal', 'Pneu.']

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

# ---------------- confusion matrices: 7 re-split models + original-split VGG16
order = sorted(SM, key=lambda k: -SM[k]['acc'])
fig, axs = plt.subplots(2, 4, figsize=(13, 7.0), dpi=220)
for ax, k in zip(axs.flat, order):
    cm_ax(ax, k, SM[k], k == HYB, " †" if SM[k].get('untraceable') else "")
k = "VGG16-10 (original split)"; cm_ax(axs.flat[7], "VGG16-10, original split ‡", SO[k])
for ax in axs[:, 0]: ax.set_ylabel('Actual class', fontsize=9)
for ax in axs[1, :]: ax.set_xlabel('Predicted class', fontsize=9)
fig.text(0.5, 0.008, 'Re-split test set: Normal n = 238, Pneumonia n = 642. Percentages are row-normalised. † Earlier run; the supplied notebook copy was interrupted. '
         '‡ Original Kermany test partition (234 / 390), shown for reference.', ha='center', fontsize=8.0, color='#555')
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.subplots_adjust(hspace=0.45); plt.savefig(OUT + 'pn_confusion_panel.png', facecolor='white'); plt.close()

# ---------------- operating points + accuracy/balanced accuracy
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.2), dpi=220, gridspec_kw={'width_ratios': [1, 1.15]})
OFF = {HYB: (8, -4), "DenseNet121-20": (-8, 10), "InceptionV3-10": (8, 4), "MobileNetV2-20": (8, -12), "VGG16-20": (8, 6), "VGG16-10": (8, -10), "CNN-A-20": (8, 4)}
for k, d in SM.items():
    c = C2 if k == HYB else C1
    x, y = (1 - d['spec']) * 100, d['sens'] * 100
    a1.errorbar(x, y, xerr=[[x - (1 - d['spec_hi']) * 100], [(1 - d['spec_lo']) * 100 - x]], yerr=[[y - d['sens_lo'] * 100], [d['sens_hi'] * 100 - y]],
                fmt='none', ecolor='#b5b5b5', elinewidth=0.9, zorder=1)
    a1.scatter(x, y, s=80 if k == HYB else 48, color=c, edgecolor='white', lw=1.3, zorder=3)
    o = OFF.get(k, (8, 4))
    a1.annotate(k.replace(" (proposed)", ""), (x, y), xytext=o, textcoords='offset points', ha='right' if o[0] < 0 else 'left', fontsize=8.2,
                color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a1.set_xlim(-0.5, 10.5); a1.set_ylim(72, 101); a1.set_xlabel('False-positive rate = 100 − specificity (%)', fontsize=9.5); a1.set_ylabel('Sensitivity (%)', fontsize=9.5)
a1.grid(True, color='#ececec'); a1.set_axisbelow(True); a1.set_title('(a) Operating points (Wilson 95% CIs)', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
models = sorted(SM, key=lambda k: SM[k]['acc']); y = np.arange(len(models))
for yi, k in zip(y, models):
    d = SM[k]; c = C2 if k == HYB else C1
    a2.plot([d['lo'] * 100, d['hi'] * 100], [yi + 0.14] * 2, color=c, lw=1.6, solid_capstyle='round')
    a2.scatter(d['acc'] * 100, yi + 0.14, s=60 if k == HYB else 40, color=c, edgecolor='white', lw=1.2, zorder=3, label='Accuracy' if yi == 0 else None)
    a2.scatter(d['bacc'] * 100, yi - 0.16, s=46 if k == HYB else 34, marker='D', color='white', edgecolor=c, lw=1.6, zorder=3, label='Balanced accuracy' if yi == 0 else None)
    a2.text(100.6, yi, f"{d['acc']*100:.2f} / {d['bacc']*100:.2f}", va='center', fontsize=8.2, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a2.set_yticks(y); a2.set_yticklabels([k.replace(" (proposed)", "") for k in models], fontsize=9)
for lbl, k in zip(a2.get_yticklabels(), models):
    if k == HYB: lbl.set_weight('bold'); lbl.set_color('#1F3864')
a2.set_xlim(77, 106); a2.xaxis.set_major_locator(FixedLocator([80, 85, 90, 95, 100])); a2.set_xlabel('Percentage (accuracy with Wilson 95% CI)', fontsize=9.5)
a2.xaxis.grid(True, color='#e6e6e6'); a2.set_axisbelow(True); a2.legend(frameon=False, fontsize=8.6, loc='upper left')
a2.set_title('(b) Accuracy and balanced accuracy', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
plt.tight_layout(); plt.savefig(OUT + 'pn_operating_points.png', facecolor='white'); plt.close()

# ---------------- HED-XAI-PN training dynamics (parsed from the notebook log)
import os
A = "/mnt/user-data/attachments/00294aa0-4669-5e77-944b-0b0344daf709/"
f = [x for x in os.listdir(A) if x.startswith("f057bdce")][0]; nb = json.load(open(A + f))
t = "".join("".join(o.get("text", "")) for o in nb["cells"][0]["outputs"] if o.get("output_type") == "stream")
t = re.sub(r"\x1b\[[0-9;]*m", "", t)
rows = re.findall(r"Epoch (\d+)/50\n.*?accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+) - learning_rate: ([\d.e+-]+)", t, flags=re.S)
ep = np.array(rows, dtype=float); e = ep[:, 0]
json.dump({'epoch': e.tolist(), 'acc': ep[:, 1].tolist(), 'loss': ep[:, 2].tolist(), 'val_acc': ep[:, 3].tolist(), 'val_loss': ep[:, 4].tolist(), 'lr': ep[:, 5].tolist()},
          open(OUT + 'hedxai_pn_history_50ep.json', 'w'), indent=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), dpi=220)
for ax, tr, va, yl, ttl in [(a1, ep[:, 1] * 100, ep[:, 3] * 100, 'Accuracy (%)', '(a) Accuracy'), (a2, ep[:, 2], ep[:, 4], 'Cross-entropy loss', '(b) Loss')]:
    ax.plot(e, tr, color=C1, lw=1.8, marker='o', ms=3, label='Training')
    ax.plot(e, va, color=C2, lw=1.8, marker='s', ms=3, label='Validation')
    ax.axvline(6, color='#555', ls='--', lw=1); ax.set_xlabel('Epoch', fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5)
    ax.axvspan(14.5, 50.5, color='#f2f2f2', zorder=0)
    ax.grid(True, color='#ededed'); ax.set_axisbelow(True); ax.set_title(ttl, loc='left', fontsize=10.5, weight='bold', color='#1F3864')
    ax.set_xlim(0, 51)
a1.set_ylim(70, 100.5); a1.legend(frameon=False, fontsize=9, loc='lower right')
a1.annotate('best val_loss (epoch 6)\n→ checkpoint', xy=(6, 96.8), xytext=(9, 78), fontsize=8.3, color='#333', arrowprops=dict(arrowstyle='-', color='#777', lw=0.8))
a1.text(32, 88, 'learning rate at its floor (1e-7)\nfrom epoch 15 onwards', ha='center', fontsize=8.2, color='#666')
a2.set_ylim(0, 1.85)
plt.tight_layout(); plt.savefig(OUT + 'pn_training_curves.png', facecolor='white'); plt.close()

# ---------------- efficiency trade-off
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=220)
OFF = {HYB: (8, -3), "DenseNet121-20": (8, -12), "InceptionV3-10": (-8, 6), "MobileNetV2-20": (8, -3), "VGG16-20": (8, 3), "VGG16-10": (8, -9), "CNN-A-20": (8, -3)}
for k, d in SM.items():
    c = C2 if k == HYB else C1
    a1.scatter(d['params'] / 1e6, d['acc'] * 100, s=80 if k == HYB else 48, color=c, edgecolor='white', lw=1.3, zorder=3)
    o = OFF[k]; a1.annotate(k.replace(" (proposed)", "") + (" †" if d.get('untraceable') else ""), (d['params'] / 1e6, d['acc'] * 100), xytext=o, textcoords='offset points',
                           ha='right' if o[0] < 0 else 'left', fontsize=8.3, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a1.set_xscale('log'); a1.set_xlim(2, 40); a1.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 40])); a1.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '20', '40'])); a1.xaxis.set_minor_locator(NullLocator())
a1.set_ylim(78, 100); a1.set_xlabel('Parameters (millions, log scale)', fontsize=9.5); a1.set_ylabel('Test accuracy (%)', fontsize=9.5)
a1.grid(True, color='#ededed'); a1.set_axisbelow(True); a1.set_title('(a) Accuracy versus model size', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
names = sorted([k for k in SM if SM[k]['ms']], key=lambda k: SM[k]['ms']); yy = np.arange(len(names))[::-1]
a2.barh(yy, [SM[k]['ms'] for k in names], 0.55, color=[C2 if k == HYB else C1 for k in names])
for yi, k in zip(yy, names):
    a2.text(SM[k]['ms'] + 4, yi, f"{SM[k]['ms']:.2f} ms", va='center', fontsize=8.5, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a2.set_yticks(yy); a2.set_yticklabels([k.replace(" (proposed)", "") + (" †" if SM[k].get('untraceable') else "") for k in names], fontsize=9); a2.set_xlim(0, 340)
a2.set_xlabel('Inference time per image (ms)', fontsize=9.5); a2.xaxis.grid(True, color='#ededed'); a2.set_axisbelow(True)
a2.set_title('(b) Inference time', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
plt.tight_layout(); plt.savefig(OUT + 'pn_tradeoff.png', facecolor='white'); plt.close()
print('ok', len(rows))
