"""Test không cần GPU / dữ liệu BTC / model:  python tests/test_unit.py   (hoặc pytest tests/)

Gồm: parse output LLM, chặn ID bịa, lịch tham chiếu, giờ am/pm, canon WER, fallback hotword, scorer, và một lần chạy
scripts/run.py --text --llm gold từ đầu tới cuối trên một split giả (zaic_tools giả) -> điểm phải là 1.0."""
import json
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lifemate.agent.loop import run_agent  # noqa: E402
from lifemate.agent.messages import date_table, parse_calls, parse_verdict, time_hints, unseen_ids  # noqa: E402
from lifemate.asr.decode import decode_turns  # noqa: E402
from lifemate.data import select  # noqa: E402
from lifemate.score import call_f1, match  # noqa: E402
from lifemate.text import canon, is_heldout, norm, wer_counts  # noqa: E402


def test_parse():
    assert parse_calls('ok <tool_call>{"name": "a.b", "arguments": {"x": 1}}</tool_call>'
                       '<tool_call>{"name": "c.d", "arguments": "{\\"y\\": 2}"}</tool_call><tool_call>bad</tool_call>') == \
        [{"name": "a.b", "arguments": {"x": 1}}, {"name": "c.d", "arguments": {"y": 2}}]
    assert parse_verdict('{"verdict": "ACCEPT"}') == (True, "")
    assert parse_verdict('```json\n{"verdict": "REJECT", "reason": "thiếu date"}\n```') == (False, "thiếu date")
    assert parse_verdict("REJECT vì sai tool")[0] is False
    assert parse_verdict("garbage")[0] is True


def test_unseen_ids():
    cat = {"x.set": {"parameters": {"properties": {"device_id": {"x-ref": "x.devices"}, "power": {}}}}}
    msgs = [{"role": "tool", "content": '{"devices": [{"id": "light_1"}]}'}]
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light_1", "power": "on"}}, cat, msgs) == []
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light_2"}}, cat, msgs) == ["device_id"]
    assert unseen_ids({"name": "x.set", "arguments": {"device_id": "light"}}, cat, msgs) == ["device_id"]  # không khớp tiền tố


def test_dates_and_times():
    t = date_table("2026-11-28T07:10:00+07:00")  # thứ bảy
    assert "- 2026-11-28: thứ bảy tuần này (hôm nay, 07:10)" in t and "(hôm nay)" in date_table("2026-11-28")
    assert "- 2026-11-29: chủ nhật tuần này (mai)" in t and "- 2026-11-30: thứ hai tuần sau" in t
    assert "- 2026-12-08: thứ ba tuần sau nữa / tuần kia" in t and "2026-12-14" not in t
    for s, want in [("đặt báo thức 9am mai", '- "9am" = 09:00'), ("hẹn 8pm tối nay", '- "8pm" = 20:00'),
                    ("họp 8:30pm", '- "8:30pm" = 20:30'), ("12am", '- "12am" = 00:00'), ("12pm", '- "12pm" = 12:00'),
                    ("báo thức 5am nhé à không 5:30am mai", '- "5:30am" = 05:30'), ("bật đèn", ""), ("13pm", "")]:
        assert time_hints(s) == want, (s, time_hints(s))


def test_agent_blocks_invented_ids():
    class Env:
        def __init__(self):
            self.calls = []

        def validate(self, name, args):
            return {"ok": True}

        def call(self, name, args):
            self.calls.append(args)
            return {"ok": True}

    class LLM:  # đề xuất một call rồi kết thúc
        def __init__(self, dev):
            self.outs = [f'<tool_call>{{"name": "x.set", "arguments": {{"device_id": "{dev}"}}}}</tool_call>', "Xong."]

        def chat(self, convs, tools=None):
            return [self.outs.pop(0)]

    cat = {"x.set": {"kind": "write", "parameters": {"properties": {"device_id": {"x-ref": "x.devices"}}}}}
    seen = {"role": "tool", "content": '{"devices": [{"id": "light_1"}]}'}
    env = Env()
    run_agent(env, LLM("ghost"), [{"role": "user", "content": "bật đèn"}], [], cat, [], 0)
    assert env.calls == []  # ID bịa: không chạy
    env = Env()
    run_agent(env, LLM("light_1"), [seen, {"role": "user", "content": "bật đèn"}], [], cat, [], 0)
    assert env.calls == [{"device_id": "light_1"}]
    env, trace = Env(), []
    run_agent(env, LLM("light_1"), [seen, {"role": "user", "content": "bật đèn"}], [], cat, trace, 0, deadline=0)
    assert env.calls == [] and trace == [{"error": "time limit"}]


def test_text():
    assert norm("Giá 2.5 triệu, nhé!") == "giá 2.5 triệu nhé" and norm("À mà thôi... cứ để; SJC") == "à mà thôi cứ để sjc"
    assert canon("huỷ") == canon("hủy") and canon("khoá") == canon("khóa") and canon("dùm") == canon("giùm")
    assert canon("hủy") != canon("huy") and canon("ma") != canon("mà")
    assert wer_counts("huỷ đơn giùm", "hủy đơn dùm") == (0, 3)
    assert wer_counts("huỷ đơn", "hủy đơn", canonical=False) == (1, 2)
    assert wer_counts("a b c", "a x c") == (1, 3) and wer_counts("a b", "") == (2, 2)
    assert is_heldout("train_0019") and not is_heldout("train_0018")


def test_bias_fallback():
    class ASR:  # có hotword thì câu bị cụt còn 1 từ
        def transcribe(self, paths, hotwords=None, words=False):
            return [[("zalo", 1.0)] if hotwords else [(w, 1.0) for w in "nhắn zalo cho lan nhé".split()] for _ in paths]

    texts, fell = decode_turns(ASR(), ["a.wav", "b.wav"], [["Zalo"], []], fallback=0.6)
    assert texts == ["nhắn zalo cho lan nhé"] * 2 and fell == [True, False]
    texts, fell = decode_turns(ASR(), ["a.wav"], [["Zalo"]], fallback=0)
    assert texts == ["zalo"] and fell == [False]


def test_score_rules():
    cat = {"m.send": {"parameters": {"properties": {"to": {}, "text": {"x-match": "fuzzy"}}}}}
    g = {"name": "m.send", "arguments": {"to": "c1", "text": "anh đến muộn mười phút"}}
    assert match({"name": "m.send", "arguments": {"to": "c1", "text": "anh đến muộn 10 phút"}}, g, cat)  # fuzzy F1 > 0.6
    assert not match({"name": "m.send", "arguments": {"to": "c2", "text": g["arguments"]["text"]}}, g, cat)
    assert not match({"name": "m.send", "arguments": {"to": "c1"}}, g, cat)  # thiếu tham số
    assert call_f1([], [], cat) == 1.0 and call_f1([g], [], cat) == 0.0
    assert call_f1([g, g], [g], cat) == 2 / 3  # call thừa giảm precision


def test_select(tmp=None):
    items = [{"id": f"train_{i:04d}"} for i in range(30)]
    assert [it["id"] for it in select(items, heldout=True)] == ["train_0009", "train_0019", "train_0029"]
    assert [it["id"] for it in select(items, ids="train_0001,train_0003_u2")] == ["train_0001", "train_0003"]
    assert len(select(items, limit=5)) == 5


FAKE_TOOLS = textwrap.dedent('''
    """ToolEnv giả cho test: state = list call ghi đã chạy; hash = hash của list đó."""
    import hashlib, json

    class ToolEnv:
        def __init__(self, split, sid, root=None):
            self.calls, self.state = [], []

        def validate(self, name, args):
            return {"ok": name in ("contacts.search", "msg.send", "sys.ask_missing_info")} | (
                {} if name in ("contacts.search", "msg.send", "sys.ask_missing_info") else {"error": "unknown tool"})

        def call(self, name, args):
            self.calls.append({"name": name, "arguments": args})
            if name == "contacts.search":
                return {"ok": True, "contacts": [{"id": "c_lan", "name": "Lan"}]}
            self.state.append([name, args])
            return {"ok": True}

        def get_tool_calls(self):
            return list(self.calls)

        def get_state_hash(self):
            return hashlib.md5(json.dumps(self.state, sort_keys=True).encode()).hexdigest()
''')


def make_fake_split(root):
    d = Path(root) / "train"
    (d / "zaic_tools").mkdir(parents=True)
    (d / "zaic_tools" / "__init__.py").write_text(FAKE_TOOLS, encoding="utf-8")
    tools = [{"name": "contacts.search", "kind": "read", "description": "tìm liên hệ",
              "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
             {"name": "msg.send", "kind": "write", "description": "gửi tin",
              "parameters": {"type": "object", "properties": {"contact_id": {"type": "string", "x-ref": "contacts"},
                                                              "text": {"type": "string", "x-match": "fuzzy"}}}}]
    (d / "train_tools_catalog.json").write_text(json.dumps({"apps": [{"name": "msg", "tools": tools}]}), encoding="utf-8")
    ctx = {"current_time": "2026-11-11T08:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh", "location": None,
           "user_profile": {"name": "Minh"}, "installed_apps": {"msg": {"tools": ["contacts.search", "msg.send"]}}}
    items = [
        {"id": "train_0001", "context": ctx, "history": [], "audio": "a1.wav", "audio_text": "Nhắn Lan là anh đến muộn",
         "label": {"tool_calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}},
                                  {"name": "msg.send", "arguments": {"contact_id": "c_lan", "text": "anh đến muộn"}}]}},
        {"id": "train_0002", "context": ctx, "history": [], "audio": "a2.wav", "audio_text": "Cảm ơn nhé",
         "label": {"tool_calls": []}},
    ]
    (d / "train.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")


def test_end_to_end_gold(capsys=None):
    import contextlib
    import io
    from scripts.run import main as run_main
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_split(tmp)
        out = Path(tmp) / "sub.jsonl"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_main(["--root", tmp, "--split", "train", "--text", "--llm", "gold", "--mapper", "none", "--out", str(out)])
        log = buf.getvalue()
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in rows] == ["train_0001", "train_0002"]
        assert len(rows[0]["tool_calls"]) == 2 and rows[1]["tool_calls"] == []
        assert "SCORE 1.0000 trên 2 mẫu" in log, log
        assert (Path(tmp) / "sub.jsonl.trace.jsonl").exists()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
