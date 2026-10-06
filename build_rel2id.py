import json
import sys

info_path = sys.argv[1]
out_path = sys.argv[2]

with open(info_path, encoding="utf-8") as f:
    info = json.load(f)

rels = sorted(info.keys())
rel2id = {"Na": 0}
for i, r in enumerate(rels):
    rel2id[r] = i + 1

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(rel2id, f, ensure_ascii=False, indent=2)

print(f"wrote {out_path}: {len(rel2id)} classes (Na=0, {len(rels)} relations)")
