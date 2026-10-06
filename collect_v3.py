import os, re, statistics, math, glob

BASE = os.path.expanduser("~/docred_project/optimized/logs_v3")
SEEDS = [42, 123, 456, 789, 2024]
CONFIGS = ["base", "ms", "gr", "cb", "msgr", "mscb", "grcb", "full"]

RE_DEV = re.compile(r"'dev_rel': \[([0-9.]+), ([0-9.]+), ([0-9.]+)\], 'dev_rel_ign': \[([0-9.]+), ([0-9.]+), ([0-9.]+)\]")
RE_EVI = re.compile(r"'dev_evi': \[([0-9.]+), ([0-9.]+), ([0-9.]+)\]")

def read_scores(path):
    if not os.path.exists(path):
        return None
    f1 = ign = evi = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = RE_DEV.search(line)
            if m:
                f1 = float(m.group(3)); ign = float(m.group(6))
            m2 = RE_EVI.search(line)
            if m2:
                evi = float(m2.group(3))
    return (f1, ign, evi)

def collect(prefix):
    out = {}
    for cfg in CONFIGS:
        f1s, igns, evis = [], [], []
        for s in SEEDS:
            p = os.path.join(BASE, f"{prefix}_{cfg}_s{s}.log")
            r = read_scores(p)
            if r and r[0] is not None:
                f1s.append(r[0]); igns.append(r[1]); evis.append(r[2])
        out[cfg] = (f1s, igns, evis)
    return out

def paired_ttest(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None, None, None
    d = [x - y for x, y in zip(a, b)]
    md = statistics.mean(d)
    sd = statistics.stdev(d)
    if sd == 0:
        return float("inf"), md, None
    t = md / (sd / math.sqrt(len(d)))
    df = len(d) - 1
    p = None
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df))
    except Exception:
        p = None
    return t, md, p

def report(prefix, label):
    data = collect(prefix)
    print(f"\n===== {label} ({prefix}_*) =====")
    base_f1 = data.get("base", ([], [], []))[0]
    for cfg in CONFIGS:
        f1s, igns, evis = data[cfg]
        if not f1s:
            print(f"{cfg:5s}: (no logs)")
            continue
        mf = statistics.mean(f1s)
        sf = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        mi = statistics.mean(igns)
        line = f"{cfg:5s}: F1 {mf:.2f} +/- {sf:.2f}  IgnF1 {mi:.2f}  n={len(f1s)}"
        if cfg != "base" and base_f1 and len(base_f1) == len(f1s):
            t, md, p = paired_ttest(f1s, base_f1)
            line += f"  dF1={md:+.2f}  t={t:.2f}"
            if p is not None:
                line += f"  p={p:.3f}"
        print(line)
    print("Per-seed F1:")
    for cfg in CONFIGS:
        f1s, _, _ = data[cfg]
        if f1s:
            print(f"  {cfg:5s}: {[f'{x:.2f}' for x in f1s]}")

if __name__ == "__main__":
    report("bert", "BERT-base")
    if glob.glob(os.path.join(BASE, "rob_*.log")):
        report("rob", "RoBERTa-base")
    if glob.glob(os.path.join(BASE, "robl_*.log")):
        report("robl", "RoBERTa-large")
    if glob.glob(os.path.join(BASE, "redoc_*.log")):
        report("redoc", "Re-DocRED")
    if glob.glob(os.path.join(BASE, "sens_*.log")):
        print("\n===== sensitivity (seed 42, full) =====")
        for f in sorted(glob.glob(os.path.join(BASE, "sens_*.log"))):
            r = read_scores(f)
            print(f"  {os.path.basename(f)}: F1={r[0]:.2f} IgnF1={r[1]:.2f}" if r and r[0] is not None else f"  {os.path.basename(f)}: (no score)")
