import json
import os

import yaml


def load_config(path):
    """YAML config; ctc.calibration_file (T_vi, T_en, threshold) overrides the defaults when it exists."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    calib = cfg["ctc"].get("calibration_file")
    for cand in (calib, os.path.join(os.path.dirname(os.path.abspath(path)), os.path.basename(calib or ""))):
        if cand and os.path.exists(cand):
            c = json.load(open(cand, encoding="utf-8"))
            for src, dst in (("T_vi", "temperature_vi"), ("T_en", "temperature_en"), ("conf_threshold", "conf_threshold")):
                if src in c:
                    cfg["ctc"][dst] = c[src]
            break
    _fallback_cpu(cfg)
    return cfg


def _fallback_cpu(cfg):
    """No CUDA on this machine: every cuda:* device in models.* becomes cpu."""
    import torch
    if torch.cuda.is_available():
        return
    for k, v in cfg["models"].items():
        if k.endswith("_device") and str(v).startswith("cuda"):
            print(f"[config] CUDA not available -> {k}: {v} -> cpu")
            cfg["models"][k] = "cpu"
