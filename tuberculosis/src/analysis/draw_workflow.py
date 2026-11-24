"""Run from the tuberculosis/ folder (analysis of the recorded results; paths may need adjusting)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family'] = 'DejaVu Sans'
NAVY = '#1F3864'
fig, ax = plt.subplots(figsize=(13, 8.4), dpi=220); ax.set_xlim(0, 130); ax.set_ylim(0, 84); ax.axis('off')

def box(x, y, w, h, num, title, body, fc, ec, fs=7.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.3", fc=fc, ec=ec, lw=1.4))
    ax.add_patch(plt.Circle((x + 2.6, y + h - 2.7), 1.9, color=ec, zorder=3))
    ax.text(x + 2.6, y + h - 2.75, str(num), ha='center', va='center', color='white', fontsize=8.5, weight='bold', zorder=4)
    ax.text(x + 5.2, y + h - 2.7, title, ha='left', va='center', fontsize=9.2, weight='bold', color='#111')
    ax.text(x + 1.6, y + h - 5.6, body, ha='left', va='top', fontsize=fs, color='#222', linespacing=1.32)

def arr(x1, y1, x2, y2, cs="arc3", c=NAVY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13, color=c, lw=1.6, connectionstyle=cs))

ax.text(65, 82, "Experimental workflow of the tuberculosis study", ha='center', fontsize=13, weight='bold', color=NAVY)
for y, lab, c in [(59, 'DATA', "#2E75B6"), (31, 'MODELLING', "#548235"), (3, 'EVALUATION &\nEXPLANATION', "#C55A11")]:
    ax.add_patch(FancyBboxPatch((0.5, y), 7, 21, boxstyle="round,pad=0.2,rounding_size=1", fc=c, ec=c))
    ax.text(4, y + 10.5, lab, rotation=90, ha='center', va='center', color='white', fontsize=8.5, weight='bold')
B = '#DEEBF7'; G = '#E2EFDA'; O = '#FBE5D6'; GR = '#EDEDED'
# ---- data row
box(10, 59, 27, 21, 1, 'Source data', 'TB Chest Radiography Database\n[91]: 3,500 Normal + 3,500 TB\nfrontal CXRs pooled from NLM,\nBelarus, NIAID and RSNA sources', B, '#2E75B6')
box(41, 59, 27, 21, 2, 'Integrity audit', 'SHA-256 hash of every file;\n330 byte-identical duplicates,\nall in the TB class\n→ 6,670 unique images\n(3,500 Normal / 3,170 TB)', B, '#2E75B6')
box(72, 59, 57, 21, 3, 'Two evaluation protocols (stratified 70/15/15, seed 42)', '', B, '#2E75B6')
ax.text(73.6, 73.4, 'Protocol A — exploratory', fontsize=8, weight='bold', color='#7F7F7F', va='top')
ax.text(73.6, 70.4, 'original 7,000 images\n(duplicates retained)\n4,900 / 1,050 / 1,050\ntest: 525 Normal / 525 TB', fontsize=7.7, va='top', linespacing=1.32, color='#444')
ax.plot([100.5, 100.5], [61, 74], color='#2E75B6', lw=0.8, ls=':')
ax.text(102, 73.4, 'Protocol B — definitive', fontsize=8, weight='bold', color='#1F4E79', va='top')
ax.text(102, 70.4, 'de-duplicated 6,670 images\n4,669 / 1,000 / 1,001\ntest: 525 Normal / 476 TB\nsplit manifest saved (CSV)', fontsize=7.7, va='top', linespacing=1.32)
arr(37.4, 69.5, 40.6, 69.5); arr(68.4, 69.5, 71.6, 69.5)
# ---- modelling row
ax.add_patch(FancyBboxPatch((10, 31), 57, 21, boxstyle="round,pad=0.3,rounding_size=1.3", fc=GR, ec='#7F7F7F', lw=1.4))
ax.add_patch(plt.Circle((12.6, 49.3), 1.9, color='#7F7F7F', zorder=3)); ax.text(12.6, 49.25, '4', ha='center', va='center', color='white', fontsize=8.5, weight='bold', zorder=4)
ax.text(15.2, 49.3, 'Protocol A benchmarks (9 runs)', fontsize=9.2, weight='bold', va='center')
ax.text(11.6, 46.2, 'Custom CNN, 128×128 grey (20 / 50 / 100 epochs)\nVGG16 (10 / 20), DenseNet201 (10), InceptionV3\n(10 / 20, 299×299), MobileNetV2 (10); /255 input\nCaveat: validation and test generators applied\nrandom augmentation; final-epoch weights', fontsize=7.7, va='top', linespacing=1.32)
ax.add_patch(FancyBboxPatch((71, 31), 58, 21, boxstyle="round,pad=0.3,rounding_size=1.3", fc=G, ec='#548235', lw=1.4))
ax.add_patch(plt.Circle((73.6, 49.3), 1.9, color='#548235', zorder=3)); ax.text(73.6, 49.25, '5', ha='center', va='center', color='white', fontsize=8.5, weight='bold', zorder=4)
ax.text(76.2, 49.3, 'Protocol B models — identical split', fontsize=9.2, weight='bold', va='center')
ax.text(72.6, 46.2, 'Benchmarks: VGG16 (clean),\nDenseNet201-20, InceptionV3-20', fontsize=7.7, va='top', linespacing=1.32)
ax.plot([98, 98], [33, 46.5], color='#548235', lw=0.8, ls=':')
ax.text(99.5, 46.2, 'HED-XAI-TB (proposed)', fontsize=8, weight='bold', color='#385723', va='top')
ax.text(99.5, 43.2, 'MobileNetV2 three-scale fusion\n+ channel attention; 2.73 M params\nlast 30 layers fine-tuned, BN frozen\nAdam 1e-4, batch 16, 10 and 20 ep;\nbest val_loss checkpoint (epoch 6)', fontsize=7.7, va='top', linespacing=1.32)
arr(86, 58.6, 86, 52.4); arr(114, 58.6, 114, 52.4)
# ---- evaluation row
box(10, 3, 37, 21, 6, 'Held-out test evaluation', 'Accuracy, sensitivity, specificity\nwith Wilson 95% CIs; precision, NPV,\nF1, MCC, κ, ROC-AUC, PR-AUC;\nconfusion matrices; test set used\nonce, after model selection', O, '#C55A11')
box(51, 3, 36, 21, 7, 'Efficiency profiling', 'Parameters, saved size (MB),\ninference time per image (three\ntiming runs), throughput,\ntraining time', O, '#C55A11')
box(91, 3, 38, 21, 8, 'Grad-CAM explanation', 'VGG16-20 benchmark (Protocol A),\nblock5_conv3 (14×14); gradient of\nthe pre-sigmoid TB logit; review of\nTB and Normal examples for lung\nlocalisation and shortcut cues', O, '#C55A11')
arr(22, 30.6, 22, 24.4)
ax.plot([100, 100, 36], [30.6, 27.4, 27.4], color=NAVY, lw=1.6); arr(36, 27.5, 36, 24.4)
arr(47.4, 13.5, 50.6, 13.5); arr(87.4, 13.5, 90.6, 13.5)
plt.savefig('results/figures/dissertation/tb_workflow.png', bbox_inches='tight', facecolor='white')
print('ok')
