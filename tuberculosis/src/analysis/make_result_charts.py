"""Run from the tuberculosis/ folder (analysis of the recorded results; paths may need adjusting)."""
import json, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#9a9a9a',
                     'axes.labelcolor': '#333', 'xtick.color': '#444', 'ytick.color': '#333'})
W = 'results/'; OUT = W + 'tbfig/'
S = json.load(open(W + 'tb_stats.json')); SB, SA = S['B'], S['A']
HYB = "HED-XAI-TB (proposed)"
C1, C2, C3, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#8c8c8c'
cmap = LinearSegmentedColormap.from_list('b', ['#f4f8fd', '#9cc3ee', '#2a78d6', '#0d3f7a'])
lab = ['Normal', 'TB']

def cm_ax(ax, k, d, hyb=False, title_extra=""):
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
    ax.set_title(f"{k}{title_extra}\nAcc {d['acc']*100:.2f}% · FN {d['fn']} · FP {d['fp']}", fontsize=9.4,
                 weight='bold' if hyb else 'normal', color='#1F3864' if hyb else '#222')
    if hyb:
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(C2); s.set_linewidth(2.5)

# ---------------- Protocol B confusion matrices
orderB = [HYB, "VGG16 (clean)", "DenseNet201-20", "InceptionV3-20"]
fig, axs = plt.subplots(1, 4, figsize=(13, 3.9), dpi=220)
for ax, k in zip(axs, orderB):
    cm_ax(ax, k, SB[k], k == HYB, "" if k == HYB else " †")
    ax.set_xlabel('Predicted class', fontsize=9)
axs[0].set_ylabel('Actual class', fontsize=9)
fig.text(0.5, 0.01, 'Protocol B test set: Normal n = 525, TB n = 476. Percentages are row-normalised. † Matrix derived from the reported precision (100%) and sensitivity.',
         ha='center', fontsize=8.4, color='#555')
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(OUT + 'tb_confusion_protocolB.png', facecolor='white'); plt.close()

# ---------------- Protocol A confusion matrices (3x3)
orderA = sorted(SA, key=lambda k: -SA[k]['acc'])
fig, axs = plt.subplots(3, 3, figsize=(10.5, 10.4), dpi=220)
for ax, k in zip(axs.flat, orderA): cm_ax(ax, k, SA[k])
for ax in axs[:, 0]: ax.set_ylabel('Actual class', fontsize=9)
for ax in axs[2, :]: ax.set_xlabel('Predicted class', fontsize=9)
fig.text(0.5, 0.008, 'Protocol A test set (duplicates retained): Normal n = 525, TB n = 525. Matrices from model.predict on the test generator.',
         ha='center', fontsize=8.4, color='#555')
plt.tight_layout(rect=(0, 0.025, 1, 1)); plt.subplots_adjust(hspace=0.42); plt.savefig(OUT + 'tb_confusion_protocolA.png', facecolor='white'); plt.close()

# ---------------- forest plot: accuracy and sensitivity with Wilson CIs, both protocols
rows = [(k, 'B') for k in sorted(SB, key=lambda k: -SB[k]['acc'])] + [(k, 'A') for k in sorted(SA, key=lambda k: -SA[k]['acc'])]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 6.4), dpi=220, sharey=True)
y = -np.arange(len(rows)).astype(float)
y[len(SB):] -= 1.0  # gap between protocols
for ax, met, lo, hi, xl, ttl in [(a1, 'acc', 'lo', 'hi', 'Accuracy (%)', '(a) Accuracy'), (a2, 'sens', 'sens_lo', 'sens_hi', 'Sensitivity for TB (%)', '(b) Sensitivity')]:
    for yi, (k, p) in zip(y, rows):
        d = (SB if p == 'B' else SA)[k]; c = C2 if k == HYB else (C1 if p == 'B' else GREY)
        ax.plot([d[lo]*100, d[hi]*100], [yi, yi], color=c, lw=1.6, solid_capstyle='round', zorder=2)
        ax.scatter(d[met]*100, yi, s=70 if k == HYB else 42, color=c, edgecolor='white', linewidth=1.2, zorder=3)
        ax.text(100.35, yi, f"{d[met]*100:.2f}", va='center', fontsize=8.2, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
    ax.set_xlim(94.5, 101.3); ax.set_xlabel(xl + ', Wilson 95% CI', fontsize=9.5)
    ax.xaxis.set_major_locator(FixedLocator([95, 96, 97, 98, 99, 100])); ax.xaxis.grid(True, color='#e6e6e6'); ax.set_axisbelow(True)
    ax.set_title(ttl, loc='left', fontsize=10.5, weight='bold', color='#1F3864')
    ax.axhspan(y[len(SB)-1] - 0.5, y[0] + 0.5, color='#eef4fb', zorder=0); ax.set_ylim(y[-1] - 0.7, y[0] + 1.2)
a1.set_yticks(y); a1.set_yticklabels([f"{k}" for k, p in rows], fontsize=9)
for lblt, (k, p) in zip(a1.get_yticklabels(), rows):
    if k == HYB: lblt.set_weight('bold'); lblt.set_color('#1F3864')
    elif p == 'A': lblt.set_color('#555')
a1.text(94.6, y[0] + 0.55, 'Protocol B — de-duplicated, n = 1,001', fontsize=8.6, weight='bold', color='#1F4E79', va='bottom')
a1.text(94.6, y[len(SB)] + 0.55, 'Protocol A — duplicates retained, n = 1,050 (exploratory)', fontsize=8.6, weight='bold', color='#555', va='bottom')
plt.tight_layout(); plt.savefig(OUT + 'tb_forest.png', facecolor='white'); plt.close()

# ---------------- HED-XAI-TB training dynamics (10-epoch run, parsed from the notebook log)
txt = open(W + 'tb/958c80ff.txt').read()
ep = re.findall(r"accuracy: ([\d.]+) - auc: [\d.]+ - loss: ([\d.]+) .*?val_accuracy: ([\d.]+) - val_auc: [\d.]+ - val_loss: ([\d.]+) .*?learning_rate: ([\d.e+-]+)", txt)[:10]
ep = np.array(ep, dtype=float); e = np.arange(1, 11)
json.dump({'epoch': e.tolist(), 'acc': ep[:, 0].tolist(), 'loss': ep[:, 1].tolist(), 'val_acc': ep[:, 2].tolist(), 'val_loss': ep[:, 3].tolist(), 'lr': ep[:, 4].tolist()},
          open(OUT + 'hedxai_tb_history_10ep.json', 'w'), indent=1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), dpi=220)
for ax, tr, va, yl, ttl in [(a1, ep[:, 0]*100, ep[:, 2]*100, 'Accuracy (%)', '(a) Accuracy'), (a2, ep[:, 1], ep[:, 3], 'Binary cross-entropy loss', '(b) Loss')]:
    ax.plot(e, tr, color=C1, lw=2, marker='o', ms=4.5, label='Training')
    ax.plot(e, va, color=C2, lw=2, marker='s', ms=4.5, label='Validation')
    ax.axvline(6, color='#555', ls='--', lw=1); ax.set_xticks(e); ax.set_xlabel('Epoch', fontsize=9.5); ax.set_ylabel(yl, fontsize=9.5)
    for x0 in (5.5, 8.5): ax.axvspan(x0, x0 + 0.04, color='#bbbbbb')
    ax.grid(True, color='#ededed'); ax.set_axisbelow(True); ax.set_title(ttl, loc='left', fontsize=10.5, weight='bold', color='#1F3864')
a1.set_ylim(84, 100.5); a1.legend(frameon=False, fontsize=9, loc='lower right')
a1.annotate('best val_loss\n(epoch 6) → checkpoint', xy=(6, 99.5), xytext=(6.4, 89), fontsize=8.3, color='#333', arrowprops=dict(arrowstyle='-', color='#777', lw=0.8))
a2.text(5.55, 0.53, 'LR ×0.2', fontsize=7.6, color='#666', rotation=90, va='top'); a2.text(8.55, 0.53, 'LR ×0.2', fontsize=7.6, color='#666', rotation=90, va='top')
plt.tight_layout(); plt.savefig(OUT + 'tb_training_curves.png', facecolor='white'); plt.close()

# ---------------- efficiency trade-off (Protocol B)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=220)
for k, d in SB.items():
    c = C2 if k == HYB else C1
    a1.scatter(d['params']/1e6, d['acc']*100, s=80 if k == HYB else 50, color=c, edgecolor='white', lw=1.3, zorder=3)
    a1.annotate(k, (d['params']/1e6, d['acc']*100), xytext=(7, -3 if k != "DenseNet201-20" else -12), textcoords='offset points', fontsize=8.4,
                color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
a1.set_xscale('log'); a1.set_xlim(2, 40); a1.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 40])); a1.xaxis.set_major_formatter(FixedFormatter(['2', '5', '10', '20', '40'])); a1.xaxis.set_minor_locator(NullLocator())
a1.set_ylim(99.2, 99.8); a1.set_xlabel('Parameters (millions, log scale)', fontsize=9.5); a1.set_ylabel('Test accuracy (%)', fontsize=9.5)
a1.grid(True, color='#ededed'); a1.set_axisbelow(True); a1.set_title('(a) Accuracy versus model size', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
names = [HYB, "DenseNet201-20", "VGG16 (clean)"]; vals = [SB[k]['ms'] for k in names]
yy = np.arange(len(names))[::-1]
a2.barh(yy, vals, 0.5, color=[C2, C1, C1])
for yi, v, k in zip(yy, vals, names):
    a2.text((max(SB[HYB]['ms_runs']) if k == HYB else v) + 1.5, yi, f"{v:.2f} ms", va='center', fontsize=8.6, color='#1F3864' if k == HYB else '#333', weight='bold' if k == HYB else 'normal')
h = SB[HYB]['ms_runs']; a2.plot([min(h), max(h)], [yy[0], yy[0]], color='#333', lw=1.2); a2.plot([min(h)]*2, [yy[0]-0.12, yy[0]+0.12], color='#333', lw=1.2); a2.plot([max(h)]*2, [yy[0]-0.12, yy[0]+0.12], color='#333', lw=1.2)
a2.set_yticks(yy); a2.set_yticklabels(names, fontsize=9); a2.set_xlim(0, 100); a2.set_xlabel('Inference time per image (ms)', fontsize=9.5)
a2.xaxis.grid(True, color='#ededed'); a2.set_axisbelow(True); a2.set_title('(b) Inference time', loc='left', fontsize=10.5, weight='bold', color='#1F3864')
plt.tight_layout(); plt.savefig(OUT + 'tb_tradeoff.png', facecolor='white'); plt.close()
print('ok')
