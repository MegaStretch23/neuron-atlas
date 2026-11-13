"""Activation providers. HFProvider needs the optional ``model`` extra (torch + transformers, CPU is enough)."""
from __future__ import annotations

import zlib


def find_act_module(model, layer: int):
    """Return the module whose output is the MLP neuron activations (after the nonlinearity) of ``layer``.

    GPT-NeoX / Pythia: model.gpt_neox.layers[i].mlp.act. GPT-2 family: model.transformer.h[i].mlp.act."""
    for root, blocks in (("gpt_neox", "layers"), ("transformer", "h")):
        r = getattr(model, root, None)
        if r is not None and hasattr(r, blocks):
            seq = getattr(r, blocks)
            if not 0 <= layer < len(seq):
                raise ValueError(f"layer {layer} out of range: the model has {len(seq)} layers (0..{len(seq) - 1})")
            mlp = getattr(seq[layer], "mlp", None)
            act = getattr(mlp, "act", None)
            if act is None:
                raise ValueError(f"layer {layer} has no mlp.act module to hook")
            return act
    raise ValueError("unsupported architecture: expected a GPT-NeoX/Pythia (gpt_neox.layers) or GPT-2 (transformer.h) model")


class HFProvider:
    def __init__(self, model_name: str, max_length: int = 128):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise SystemExit("the real-model path needs torch and transformers: pip install 'neuron-atlas[model]'") from e
        self.name = model_name
        self.max_length = max_length
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def run(self, text, layer):
        torch = self._torch
        act = find_act_module(self.model, layer)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        got = {}
        hook = act.register_forward_hook(lambda _m, _i, out: got.__setitem__("a", out.detach()))
        try:
            with torch.no_grad():
                self.model(**enc)
        finally:
            hook.remove()
        ids = enc["input_ids"][0].tolist()
        return [self.tok.decode([i], clean_up_tokenization_spaces=False) for i in ids], got["a"][0].tolist()


class ToyProvider:
    """Deterministic stand-in with no download, for demos and tests. NOT a language model: each 'neuron' fires
    on words whose hash falls in its bucket, so its atlas shows the plumbing works, not anything about a model."""
    name = "toy (hash buckets, not a language model)"

    def __init__(self, n_neurons: int = 16):
        self.n = n_neurons

    def run(self, text, layer):
        words = text.split(" ")
        tokens = [w if i == 0 else " " + w for i, w in enumerate(words)]
        acts = []
        for w in words:
            h = zlib.crc32(w.lower().strip(".,!?").encode()) % self.n
            acts.append([1.0 if k == h else 0.0 for k in range(self.n)])
        return tokens, acts
