from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .atlas import build_atlas, read_corpus
from .html import render_html


def _provider(a):
    from .providers import HFProvider, ToyProvider
    if a.model == "toy":
        return ToyProvider()
    return HFProvider(a.model, max_length=a.max_length)


def cmd_build(a) -> int:
    corpus = Path(a.corpus)
    if not corpus.is_file():
        print(f"error: corpus file not found: {corpus}", file=sys.stderr)
        return 2
    try:
        texts = read_corpus(corpus)
        neurons = [int(x) for x in a.neurons.split(",")] if a.neurons else None
        atlas = build_atlas(_provider(a), texts, a.layer, top_k=a.top_k, context=a.context, neurons=neurons,
                            skip_first=a.skip_first, model_name=a.model)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.json").write_text(json.dumps(atlas, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "atlas.html").write_text(render_html(atlas), encoding="utf-8")
    print(f"model={atlas['model']} layer={atlas['layer']} neurons={atlas['neurons_included']}/{atlas['n_neurons']} "
          f"texts={atlas['n_texts']} tokens={atlas['n_tokens']}")
    print(f"wrote {out / 'atlas.json'} and {out / 'atlas.html'}")
    return 0


def cmd_html(a) -> int:
    try:
        atlas = json.loads(Path(a.atlas).read_text(encoding="utf-8"))
        if not isinstance(atlas, dict) or "neurons" not in atlas:
            raise ValueError("not an atlas.json (no 'neurons' list)")
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    Path(a.out).write_text(render_html(atlas), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="neuron-atlas", description="Top-activating tokens and contexts for the MLP neurons of one layer.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="run a corpus through a model and write atlas.json + atlas.html")
    b.add_argument("--model", default="EleutherAI/pythia-14m", help="Hugging Face model id (Pythia / GPT-2 family) or 'toy' for the no-download demo")
    b.add_argument("--layer", type=int, default=3)
    b.add_argument("--corpus", required=True, help="text file, one text per line")
    b.add_argument("--out", default="atlas-out")
    b.add_argument("--top-k", type=int, default=10)
    b.add_argument("--context", type=int, default=6, help="tokens of context shown on each side")
    b.add_argument("--max-length", type=int, default=128, help="truncate each text to this many tokens")
    b.add_argument("--neurons", help="comma-separated neuron ids (default: all)")
    b.add_argument("--skip-first", action="store_true", help="ignore the first token of each text")
    b.set_defaults(fn=cmd_build)
    h = sub.add_parser("html", help="re-render atlas.html from an atlas.json")
    h.add_argument("atlas")
    h.add_argument("-o", "--out", default="atlas.html")
    h.set_defaults(fn=cmd_html)
    a = ap.parse_args(argv)
    return a.fn(a)
