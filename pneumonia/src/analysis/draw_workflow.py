"""Run from the pneumonia/ folder (analysis of the recorded results; paths may need adjusting)."""
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

ax.text(65, 82, "Experimental workflow of the pneumonia study", ha='center', fontsize=13, weight='bold', color=NAVY)
for y, lab, c in [(59, 'DATA', "#2E75B6"), (31, 'MODELLING', "#548235"), (3, 'EVALUATION &\nEXPLANATION', "#C55A11")]:
    ax.add_patch(FancyBboxPatch((0.5, y), 7, 21, boxstyle="round,pad=0.2,rounding_size=1", fc=c, ec=c))
    ax.text(4, y + 10.5, lab, rotation=90, ha='center', va='center', color='white', fontsize=8.5, weight='bold')
B = '#DEEBF7'; G = '#E2EFDA'; O = '#FBE5D6'; GR = '#EDEDED'
box(10, 59, 27, 21, 1, 'Source data', 'Kermany paediatric CXR [92]:\n5,856 images, 1,583 Normal /\n4,273 Pneumonia; children aged\n1–5 years, Guangzhou; original\nsplit 5,216 / 16 / 624', B, '#2E75B6')
box(41, 59, 27, 21, 2, 'Pooled re-split', 'All folders pooled; each class\nshuffled and split 70/15/15\n(no random seed recorded)\nTrain 4,099 · Val 877 · Test 880\n(test: 238 Normal / 642 Pn)', B, '#2E75B6')
box(72, 59, 57, 21, 3, 'Integrity audit and data state at training time', 'SHA-256 audit (HED-XAI-PN notebook): 16 byte-identical images occur in more\nthan one partition (not removed). Validation folder read as 880 images\n(240 Normal) in later runs vs 877 at creation. Patient IDs are encoded in the\nfile names (personN_…, IM-NNNN) but were not used: image-level split.\nTest-set counts (238 / 642) identical in all reported runs.', B, '#2E75B6', fs=7.5)
arr(37.4, 69.5, 40.6, 69.5); arr(68.4, 69.5, 71.6, 69.5)
ax.add_patch(FancyBboxPatch((10, 31), 57, 21, boxstyle="round,pad=0.3,rounding_size=1.3", fc=GR, ec='#7F7F7F', lw=1.4))
ax.add_patch(plt.Circle((12.6, 49.3), 1.9, color='#7F7F7F', zorder=3)); ax.text(12.6, 49.25, '4', ha='center', va='center', color='white', fontsize=8.5, weight='bold', zorder=4)
ax.text(15.2, 49.3, 'Benchmarks (single backbone)', fontsize=9.2, weight='bold', va='center')
ax.text(11.6, 46.2, 'VGG16 (10 / 20), DenseNet121 (20), MobileNetV2 (20):\n  frozen ImageNet backbones, /255, balanced class\n  weights, Adam 1e-3, final-epoch weights\nInceptionV3 (10): 299×299, last 30 layers tuned\nCustom CNN-A (20): from scratch, 224×224', fontsize=7.7, va='top', linespacing=1.32)
ax.add_patch(FancyBboxPatch((71, 31), 58, 21, boxstyle="round,pad=0.3,rounding_size=1.3", fc=G, ec='#548235', lw=1.4))
ax.add_patch(plt.Circle((73.6, 49.3), 1.9, color='#548235', zorder=3)); ax.text(73.6, 49.25, '5', ha='center', va='center', color='white', fontsize=8.5, weight='bold', zorder=4)
ax.text(76.2, 49.3, 'Proposed model: HED-XAI-PN', fontsize=9.2, weight='bold', va='center')
ax.text(72.6, 46.2, 'MobileNetV2 three-scale fusion + channel attention + softmax (2 units);\n2,728,226 parameters; last 30 layers fine-tuned, BN frozen;\nMobileNetV2 preprocess_input; light augmentation (no flipping);\nAdam 1e-4, batch 16, 50 epochs, no class weights;\nbest val_loss checkpoint (epoch 6) evaluated once on the test set', fontsize=7.7, va='top', linespacing=1.32)
arr(54.5, 58.6, 54.5, 52.4); arr(100, 58.6, 100, 52.4)
box(10, 3, 37, 21, 6, 'Held-out test (n = 880)', 'Accuracy, sensitivity, specificity\n(Wilson 95% CIs), balanced accuracy,\nprecision, NPV, F1, macro-F1, MCC,\nκ, ROC-AUC, average precision;\nconfusion matrices; z-tests', O, '#C55A11')
box(51, 3, 36, 21, 7, 'Efficiency profiling', 'Parameters, saved size (MB),\ninference time per image over the\nfull test set, training time', O, '#C55A11')
box(91, 3, 38, 21, 8, 'Grad-CAM explanation', 'DenseNet121-20 (conv5_block16_2_conv,\n7×7) and VGG16-20 (block5_conv3,\n14×14); gradient of the predicted-class\nsigmoid score; review of correct and\nmisclassified radiographs', O, '#C55A11')
arr(22, 30.6, 22, 24.4)
ax.plot([100, 100, 36], [30.6, 27.4, 27.4], color=NAVY, lw=1.6); arr(36, 27.5, 36, 24.4)
arr(47.4, 13.5, 50.6, 13.5); arr(87.4, 13.5, 90.6, 13.5)
plt.savefig('results/figures/dissertation/pn_workflow.png', bbox_inches='tight', facecolor='white')
print('ok')
