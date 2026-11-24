"""Patient-level utilities for the Kermany pneumonia data (added after the dissertation experiments).

The Kermany file names encode a patient identifier:
    PNEUMONIA:  person<N>_bacteria_<k>.jpeg / person<N>_virus_<k>.jpeg   ->  patient "person<N>"
    NORMAL:     IM-<NNNN>-<kkkk>.jpeg / NORMAL2-IM-<NNNN>-<kkkk>.jpeg     ->  patient "IM-<NNNN>" / "NORMAL2-IM-<NNNN>"
                (the NORMAL2 prefix is kept, because it marks a separate numbering series)

Usage (run from the pneumonia/ folder):
    python src/patient_level_tools.py overlap  [data/chest_xray_new]      # how many test patients also occur in train/val
    python src/patient_level_tools.py split    [data/raw/chest_xray] [data/chest_xray_patient]   # 70/15/15 split by patient

The `split` command pools all images, assigns whole patients to one partition (seeded, stratified by class) and copies
the files. It was NOT used for the results reported in the dissertation; it implements the recommended follow-up.
"""
import os, re, sys, shutil, random, collections

CLASSES = ["NORMAL", "PNEUMONIA"]
EXT = (".jpeg", ".jpg", ".png")

def patient_id(fname):
    m = re.search(r"(person\d+)_", fname)
    if m: return m.group(1)
    m = re.search(r"((?:NORMAL2-)?IM-\d+)-\d+", fname)
    if m: return m.group(1)
    return os.path.splitext(fname)[0]          # fall back to the image itself

def scan(root):
    out = []
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            d = os.path.join(root, split, cls)
            if os.path.isdir(d):
                out += [(split, cls, os.path.join(d, f), patient_id(f)) for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
    return out

def overlap(root):
    rows = scan(root)
    pats = collections.defaultdict(set)
    for split, cls, path, pid in rows: pats[split].add((cls, pid))
    test = pats["test"]; other = pats["train"] | pats["val"]
    shared = test & other
    n_img = sum(1 for s, c, p, pid in rows if s == "test" and (c, pid) in shared)
    print(f"images: {len(rows)}; patients: {len(set((c, pid) for _, c, _, pid in rows))}")
    print(f"test patients: {len(test)}; also in train/val: {len(shared)}")
    print(f"test images belonging to those patients: {n_img} of {sum(1 for r in rows if r[0] == 'test')}")

def split(raw, dst, seed=42, frac=(0.70, 0.15, 0.15)):
    rnd = random.Random(seed)
    rows = scan(raw)
    for cls in CLASSES:
        by_pat = collections.defaultdict(list)
        for _, c, path, pid in rows:
            if c == cls: by_pat[pid].append(path)
        pids = sorted(by_pat); rnd.shuffle(pids)
        n = sum(len(by_pat[p]) for p in pids); cut1, cut2 = frac[0] * n, (frac[0] + frac[1]) * n
        acc = 0
        for pid in pids:
            part = "train" if acc < cut1 else ("val" if acc < cut2 else "test")
            acc += len(by_pat[pid])
            os.makedirs(os.path.join(dst, part, cls), exist_ok=True)
            for path in by_pat[pid]:
                shutil.copy(path, os.path.join(dst, part, cls, os.path.basename(path)))
    for part in ("train", "val", "test"):
        print(part, {c: len(os.listdir(os.path.join(dst, part, c))) for c in CLASSES if os.path.isdir(os.path.join(dst, part, c))})

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "overlap"
    if cmd == "overlap":
        overlap(sys.argv[2] if len(sys.argv) > 2 else "data/chest_xray_new")
    elif cmd == "split":
        split(sys.argv[2] if len(sys.argv) > 2 else "data/raw/chest_xray", sys.argv[3] if len(sys.argv) > 3 else "data/chest_xray_patient")
