import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "dataset/meta/rel2id.json"
with open(path, encoding="utf-8") as f:
    rel2id = json.load(f)

if not isinstance(rel2id, dict):
    if isinstance(rel2id, list) and len(rel2id) == 2 and isinstance(rel2id[1], dict):
        rel2id = rel2id[1]
    else:
        raise SystemExit("unexpected rel2id format")

has_na = any(str(k).lower() == "na" for k in rel2id)
n = len(rel2id)
print(f"len={n}, has_Na={has_na}")

if not has_na:
    new = {"Na": 0}
    for k, v in rel2id.items():
        new[k] = int(v) + 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new, f, ensure_ascii=False, indent=2)
    print(f"normalized -> added Na at 0, shifted ids; num_class = {len(new)}")
else:
    print(f"ok; num_class = {n}")
