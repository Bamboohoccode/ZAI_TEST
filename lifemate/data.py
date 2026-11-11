"""Đọc dữ liệu BTC: <root>/<split>/<split>.jsonl, catalog tool, audio; tìm package zaic_tools; chọn tập mẫu.
Không đọc <split>_environment/ (state.json chỉ ToolEnv được nạp)."""
import json
import re
import sys
from pathlib import Path

from lifemate.text import is_heldout


def load_items(root, split):
    return [json.loads(l) for l in open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") if l.strip()]


def load_catalog(root, split):
    """{tên tool: spec} của mọi app trong catalog của split."""
    cat = json.load(open(Path(root) / split / f"{split}_tools_catalog.json", encoding="utf-8"))
    return {t["name"]: t for app in cat["apps"] for t in app["tools"]}


def user_turns(item):
    """các lượt user theo thứ tự: history trước, query (chính item) cuối."""
    return [h for h in item["history"] if h["role"] == "user"] + [item]


def select(items, heldout=False, ids=None, limit=0):
    """heldout: chỉ id % 10 == 9. ids: "train_0001,train_0003" hoặc file txt mỗi dòng 1 id (id lượt "train_0008_u2"
    -> sample train_0008). limit: lấy n mẫu đầu."""
    if heldout:
        items = [it for it in items if is_heldout(it["id"])]
    if ids:
        want = Path(ids).read_text(encoding="utf-8").split() if Path(ids).is_file() else ids.split(",")
        want = {re.sub(r"_u\d+$", "", w.strip()) for w in want if w.strip()}
        items = [it for it in items if it["id"] in want]
        print(f"--ids: {len(items)}/{len(want)} sample tìm thấy")
    return items[:limit] if limit else items


def import_tool_env(root, split, tools_dir=None):
    """ToolEnv của BTC. zaic_tools có thể chỉ nằm trong train/: thử tools_dir, root/split, root/train, root."""
    root = Path(root)
    dirs = [Path(tools_dir)] if tools_dir else [root / split, root / "train", root]
    found = next((d for d in dirs if (d / "zaic_tools").is_dir() or (d / "zaic_tools.py").is_file()), None)
    if found is None:
        sys.exit(f"không tìm thấy zaic_tools trong {[str(d) for d in dirs]}; chỉ đường bằng --tools-dir")
    if str(found.resolve()) not in sys.path:
        sys.path.insert(0, str(found.resolve()))
    from zaic_tools import ToolEnv
    return ToolEnv
