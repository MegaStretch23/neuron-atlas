import json, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import neuron_atlas as na  # noqa: E402

MERGES = [("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"), ("h", "e"), ("i", "n"), ("e", "r")]
CFG = {"model_type": "gpt2", "n_layer": 2, "n_head": 2, "n_embd": 16, "n_positions": 64, "activation_function": "gelu_new",
       "layer_norm_epsilon": 1e-5, "vocab_size": 263}


def make_model(d: Path, seed=0, dtype="F32", prefix="transformer.", cfg=None):
    """A random-weight GPT-2-shaped model with a real byte-level BPE vocab (256 byte symbols + 6 merges + <|endoftext|>)."""
    cfg = dict(cfg or CFG)
    b2u = na.bytes_to_unicode()
    vocab = {b2u[b]: i for i, b in enumerate(sorted(b2u))}
    for a, b in MERGES:
        vocab[a + b] = len(vocab)
    vocab["<|endoftext|>"] = len(vocab)
    cfg.update(vocab_size=len(vocab), bos_token_id=vocab["<|endoftext|>"])
    rng = np.random.default_rng(seed)
    D, V, P, H = cfg["n_embd"], len(vocab), cfg["n_positions"], 4 * cfg["n_embd"]
    r = lambda *s: (rng.standard_normal(s) * 0.3).astype(np.float32)
    W = {"wte.weight": r(V, D), "wpe.weight": r(P, D), "ln_f.weight": 1 + r(D) * .1, "ln_f.bias": r(D) * .1}
    for i in range(cfg["n_layer"]):
        W.update({f"h.{i}.ln_1.weight": 1 + r(D) * .1, f"h.{i}.ln_1.bias": r(D) * .1, f"h.{i}.ln_2.weight": 1 + r(D) * .1, f"h.{i}.ln_2.bias": r(D) * .1,
                  f"h.{i}.attn.c_attn.weight": r(D, 3 * D), f"h.{i}.attn.c_attn.bias": r(3 * D) * .1,
                  f"h.{i}.attn.c_proj.weight": r(D, D), f"h.{i}.attn.c_proj.bias": r(D) * .1,
                  f"h.{i}.mlp.c_fc.weight": r(D, H), f"h.{i}.mlp.c_fc.bias": r(H) * .1,
                  f"h.{i}.mlp.c_proj.weight": r(H, D), f"h.{i}.mlp.c_proj.bias": r(D) * .1,
                  f"h.{i}.attn.bias": np.tril(np.ones((1, 1, P, P), np.float32))})   # buffer that real checkpoints carry; must be ignored
    d.mkdir(parents=True, exist_ok=True)
    na.write_safetensors(d / "model.safetensors", {prefix + k: v for k, v in W.items()}, dtype=dtype)
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in MERGES) + "\n")
    return d


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return make_model(tmp_path_factory.mktemp("tiny"))


DOCS = ["the cat in the hat went there in the rain.", "In the end, the answer is 42; the rest is noise.\nA new line, then more text here.",
        "Hello, world! it's 2024_x"]
