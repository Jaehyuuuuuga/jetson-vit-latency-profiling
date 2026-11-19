#!/usr/bin/env python3
"""trtexec 프로파일 JSON을 연산 종류별로 집계하고 기기 간 비교 그래프를 그린다.
사용법: python parse_profile.py Orin=results/orin/vit/profile.json Thor=results/thor/vit/profile.json -o figures/vit
"""
import argparse, json, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 위에서부터 먼저 걸리는 규칙이 적용된다. 병합된 레이어는 이름에 여러 연산이 섞이므로 대략적인 분류다.
RULES = [
    ("reformat", r"reformat|reorder"),
    ("fused subgraph", r"foreignnode"),
    ("convolution", r"conv"),
    ("attention", r"matmul|softmax|attn|attention"),
    ("normalization", r"layernorm|norm|reducemean|sqrt"),
    ("activation", r"gelu|erf|relu|silu|sigmoid|swish"),
    ("linear", r"gemm|linear|dense|\bfc\b"),
    ("shape ops", r"reshape|transpose|shuffle|slice|concat|gather|expand|squeeze|resize"),
]

def classify(name):
    low = name.lower()
    for cat, pat in RULES:
        if re.search(pat, low):
            return cat
    return "other"

def load(path):
    rows = [r for r in json.load(open(path)) if isinstance(r, dict) and "name" in r]
    df = pd.DataFrame(rows)
    col = next(c for c in ("averageMs", "medianMs", "timeMs") if c in df.columns)
    df = df.rename(columns={col: "ms"})[["name", "ms"]]
    df["category"] = df["name"].map(classify)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+", help="라벨=profile.json")
    ap.add_argument("-o", "--out", default="figures/profile")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cats, tops = {}, []
    for item in a.profiles:
        label, path = item.split("=", 1)
        df = load(path)
        cats[label] = df.groupby("category")["ms"].sum()
        t = df.nlargest(15, "ms").assign(device=label)
        t["share_%"] = (t["ms"] / df["ms"].sum() * 100).round(1)
        tops.append(t)
        print(f"{label}: 레이어 {len(df)}개, 합계 {df['ms'].sum():.2f} ms")
    table = pd.DataFrame(cats).fillna(0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    table.round(3).to_csv(f"{out}_by_category.csv")
    pd.concat(tops)[["device", "name", "category", "ms", "share_%"]].to_csv(f"{out}_top_layers.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    table.T.plot.bar(stacked=True, ax=axes[0], rot=0)
    axes[0].set_ylabel("ms"); axes[0].set_title("Time by op category")
    (table / table.sum() * 100).T.plot.bar(stacked=True, ax=axes[1], rot=0, legend=False)
    axes[1].set_ylabel("%"); axes[1].set_title("Share by op category")
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout(); fig.savefig(f"{out}_by_category.png", dpi=160)
    print(table.round(2))

if __name__ == "__main__":
    main()
