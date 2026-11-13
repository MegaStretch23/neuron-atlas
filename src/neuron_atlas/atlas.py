"""Core ranking logic. Standard library only: it never imports torch.

A *provider* is any object with ``run(text, layer) -> (tokens, acts)`` where ``tokens`` is a list of
strings and ``acts`` is a list (one row per token) of lists (one float per MLP neuron)."""
from __future__ import annotations

import heapq
import math
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1


def read_corpus(path) -> list[str]:
    """One text per line; blank lines are dropped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def _validate(texts, layer, top_k, context, neurons):
    if not texts:
        raise ValueError("corpus is empty: give at least one non-blank text")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError(f"layer must be a non-negative integer, got {layer!r}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if context < 0:
        raise ValueError("context must be 0 or more")
    if neurons is not None and (not neurons or any((not isinstance(n, int)) or n < 0 for n in neurons)):
        raise ValueError("neurons must be a non-empty list of non-negative integers")


def build_atlas(provider, texts, layer, top_k=10, context=6, neurons=None, skip_first=False, model_name=None) -> dict:
    """Run every text through the provider and keep, per neuron, the ``top_k`` highest activations.

    ``skip_first`` ignores the first token of each text (many models put very large activations there).
    Ties are broken by corpus order, so the output is deterministic."""
    texts = [t for t in texts if t and t.strip()]
    _validate(texts, layer, top_k, context, neurons)
    heaps: dict[int, list] = {}
    total: dict[int, float] = {}
    maxes: dict[int, float] = {}
    n_tokens = 0
    width = None
    all_tokens: list[list[str]] = []
    seq = 0
    for ti, text in enumerate(texts):
        tokens, acts = provider.run(text, layer)
        if len(tokens) != len(acts):
            raise ValueError(f"provider returned {len(tokens)} tokens but {len(acts)} activation rows for text {ti}")
        all_tokens.append(list(tokens))
        for pos, row in enumerate(acts):
            if width is None:
                width = len(row)
                if width == 0:
                    raise ValueError("provider returned zero neurons")
                if neurons is not None and max(neurons) >= width:
                    raise ValueError(f"neuron {max(neurons)} out of range: the layer has {width} neurons")
            elif len(row) != width:
                raise ValueError(f"ragged activations: expected {width} neurons, got {len(row)} (text {ti}, token {pos})")
            if skip_first and pos == 0:
                continue
            n_tokens += 1
            seq += 1
            for nid in (neurons if neurons is not None else range(width)):
                a = row[nid]
                if not math.isfinite(a):
                    raise ValueError(f"non-finite activation for neuron {nid} (text {ti}, token {pos})")
                total[nid] = total.get(nid, 0.0) + a
                if nid not in maxes or a > maxes[nid]:
                    maxes[nid] = a
                item = (a, -seq, ti, pos)
                h = heaps.setdefault(nid, [])
                if len(h) < top_k:
                    heapq.heappush(h, item)
                elif item > h[0]:
                    heapq.heapreplace(h, item)
    if n_tokens == 0:
        raise ValueError("no tokens to rank (every text was empty or skipped)")
    out = []
    for nid in sorted(heaps):
        ex = []
        for a, _neg, ti, pos in sorted(heaps[nid], reverse=True):
            toks = all_tokens[ti]
            ex.append({"act": round(a, 4), "text_idx": ti, "pos": pos, "token": toks[pos],
                       "before": "".join(toks[max(0, pos - context):pos]), "after": "".join(toks[pos + 1:pos + 1 + context])})
        out.append({"id": nid, "max_act": round(maxes[nid], 4), "mean_act": round(total[nid] / n_tokens, 4),
                    "top_tokens": _top_tokens(ex), "examples": ex})
    return {"schema": SCHEMA_VERSION, "model": model_name or getattr(provider, "name", "unknown"), "layer": layer,
            "n_neurons": width, "neurons_included": len(out), "n_texts": len(texts), "n_tokens": n_tokens,
            "top_k": top_k, "context": context, "skip_first": bool(skip_first), "neurons": out}


def _top_tokens(examples, limit=5):
    """Tokens that appear among a neuron's top examples: by count, then best activation, then text."""
    cnt = Counter(e["token"] for e in examples)
    best: dict[str, float] = {}
    for e in examples:
        best[e["token"]] = max(best.get(e["token"], e["act"]), e["act"])
    ranked = sorted(cnt, key=lambda t: (-cnt[t], -best[t], t))[:limit]
    return [{"token": t, "count": cnt[t], "max_act": best[t]} for t in ranked]
