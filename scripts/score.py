"""Chấm một hoặc nhiều submission trên split có nhãn (luật LifeMate, xem lifemate/score.py).

    python scripts/score.py --root DATA --split train sub_a.jsonl [sub_b.jsonl ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifemate.data import import_tool_env, load_catalog, load_items  # noqa: E402
from lifemate.score import load_jsonl, report, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="+")
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tools-dir")
    ap.add_argument("--worst", type=int, default=15, help="in ngần này mẫu điểm thấp nhất")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    ToolEnv = import_tool_env(root, a.split, a.tools_dir)
    catalog, items = load_catalog(root, a.split), load_items(root, a.split)
    for path in a.subs:
        print(f"== {path}")
        report(*score(load_jsonl(path), items, catalog, ToolEnv, a.split, root), worst=a.worst)


if __name__ == "__main__":
    main()
