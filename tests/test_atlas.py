import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neuron_atlas import build_atlas, read_corpus
from neuron_atlas.cli import main
from neuron_atlas.html import render_html
from neuron_atlas.providers import ToyProvider, find_act_module

ROOT = Path(__file__).resolve().parent.parent


class Fake:
    """Fixed activations: text -> (tokens, rows). No model, no download."""
    name = "fake"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def run(self, text, layer):
        self.calls.append((text, layer))
        return self.table[text]


def tbl():
    return {
        "a b c d": (["a", " b", " c", " d"], [[0.1, 5.0], [0.9, 1.0], [0.3, 2.0], [0.5, 4.0]]),
        "e f": (["e", " f"], [[0.9, 0.0], [0.2, 3.0]]),
    }


def atlas(**kw):
    return build_atlas(Fake(tbl()), ["a b c d", "e f"], layer=1, **kw)


def n(a, i):
    return next(x for x in a["neurons"] if x["id"] == i)


def test_top_examples_sorted_descending():
    a = atlas(top_k=3)
    assert [e["act"] for e in n(a, 0)["examples"]] == [0.9, 0.9, 0.5]
    assert [e["act"] for e in n(a, 1)["examples"]] == [5.0, 4.0, 3.0]


def test_top_k_limits_examples():
    assert len(n(atlas(top_k=2), 1)["examples"]) == 2


def test_top_k_larger_than_tokens_returns_all():
    assert len(n(atlas(top_k=50), 1)["examples"]) == 6


def test_ties_broken_by_corpus_order():
    ex = n(atlas(top_k=2), 0)["examples"]
    assert (ex[0]["text_idx"], ex[0]["pos"]) == (0, 1)
    assert (ex[1]["text_idx"], ex[1]["pos"]) == (1, 0)


def test_max_and_mean():
    a = atlas()
    assert n(a, 1)["max_act"] == 5.0
    assert n(a, 0)["mean_act"] == round((0.1 + 0.9 + 0.3 + 0.5 + 0.9 + 0.2) / 6, 4)


def test_counts_in_header():
    a = atlas()
    assert (a["n_neurons"], a["n_texts"], a["n_tokens"], a["layer"], a["model"]) == (2, 2, 6, 1, "fake")


def test_context_window_at_start_and_end():
    ex = n(atlas(context=2), 1)["examples"]
    first = next(e for e in ex if e["token"] == "a" or e["act"] == 5.0)
    assert first["token"] == "a" and first["before"] == "" and first["after"] == " b c"
    last = next(e for e in ex if e["token"] == " d")
    assert last["before"] == " b c" and last["after"] == ""


def test_context_zero():
    e = n(atlas(context=0), 1)["examples"][0]
    assert e["before"] == "" and e["after"] == ""


def test_skip_first_drops_first_token_of_each_text():
    a = atlas(skip_first=True)
    assert a["n_tokens"] == 4 and a["skip_first"] is True
    assert all(e["pos"] != 0 for x in a["neurons"] for e in x["examples"])
    assert n(a, 1)["max_act"] == 4.0


def test_neuron_subset():
    a = atlas(neurons=[1])
    assert [x["id"] for x in a["neurons"]] == [1] and a["neurons_included"] == 1 and a["n_neurons"] == 2


def test_neuron_out_of_range_rejected():
    with pytest.raises(ValueError, match="out of range"):
        atlas(neurons=[7])


def test_top_tokens_aggregate_by_count_then_activation():
    prov = Fake({"x": (["p", "q", "p", "r"], [[1.0], [3.0], [2.0], [0.5]])})
    a = build_atlas(prov, ["x"], layer=0, top_k=4)
    tt = a["neurons"][0]["top_tokens"]
    assert tt[0] == {"token": "p", "count": 2, "max_act": 2.0}
    assert tt[1]["token"] == "q"


def test_blank_texts_skipped_and_empty_corpus_rejected():
    prov = Fake(tbl())
    assert build_atlas(prov, ["", "  ", "e f"], layer=0)["n_texts"] == 1
    with pytest.raises(ValueError, match="corpus is empty"):
        build_atlas(prov, ["", " "], layer=0)


@pytest.mark.parametrize("kw", [{"top_k": 0}, {"context": -1}, {"neurons": []}, {"neurons": [-1]}])
def test_bad_parameters_rejected(kw):
    with pytest.raises(ValueError):
        atlas(**kw)


def test_negative_layer_rejected():
    with pytest.raises(ValueError, match="layer"):
        build_atlas(Fake(tbl()), ["e f"], layer=-1)


def test_layer_is_passed_to_provider():
    p = Fake(tbl())
    build_atlas(p, ["e f"], layer=4)
    assert p.calls == [("e f", 4)]


def test_ragged_rows_rejected():
    p = Fake({"x": (["a", "b"], [[1.0, 2.0], [1.0]])})
    with pytest.raises(ValueError, match="ragged"):
        build_atlas(p, ["x"], layer=0)


def test_token_row_mismatch_rejected():
    p = Fake({"x": (["a"], [[1.0], [2.0]])})
    with pytest.raises(ValueError, match="activation rows"):
        build_atlas(p, ["x"], layer=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_activation_rejected(bad):
    p = Fake({"x": (["a"], [[bad]])})
    with pytest.raises(ValueError, match="non-finite"):
        build_atlas(p, ["x"], layer=0)


def test_negative_activations_ranked_correctly():
    p = Fake({"x": (["a", "b"], [[-3.0], [-1.0]])})
    ex = build_atlas(p, ["x"], layer=0, top_k=1)["neurons"][0]["examples"]
    assert ex[0]["token"] == "b" and ex[0]["act"] == -1.0


def test_deterministic_output():
    assert json.dumps(atlas()) == json.dumps(atlas())


def test_json_roundtrip():
    a = atlas()
    assert json.loads(json.dumps(a)) == a


def test_read_corpus(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("one\n\n  two  \n\t\nthree\n", encoding="utf-8")
    assert read_corpus(f) == ["one", "two", "three"]


# ---- html ----

def test_html_embeds_atlas_and_is_offline():
    h = render_html(atlas())
    assert 'id="atlas"' in h and "neuron atlas" in h
    assert not re.search(r"(src|href)=[\"']?https?://", h) and "fetch(" not in h and "XMLHttpRequest" not in h


def test_html_escapes_script_breakers():
    p = Fake({"x": (["</script><b>", "&"], [[1.0], [2.0]])})
    h = render_html(build_atlas(p, ["x"], layer=0, model_name="m<script>"))
    assert h.count("</script>") == 2          # only the two real closing tags of the template
    assert "<b>" not in h and "m<script>" not in h
    m = re.search(r'<script type="application/json" id="atlas">(.*?)</script>', h, re.S)
    assert json.loads(m.group(1))["neurons"][0]["examples"][0]["token"] == "&"


def test_html_deterministic():
    assert render_html(atlas()) == render_html(atlas())


# ---- providers ----

def test_toy_provider_shape_and_determinism():
    t = ToyProvider(8)
    toks, acts = t.run("The cat sat.", 0)
    assert len(toks) == len(acts) == 3 and all(len(r) == 8 for r in acts)
    assert t.run("The cat sat.", 0) == (toks, acts)
    assert "".join(toks) == "The cat sat."


def test_find_act_module_neox_and_gpt2():
    leaf = object()
    neox = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(act=leaf))]))
    gpt2 = SimpleNamespace(transformer=SimpleNamespace(h=[SimpleNamespace(mlp=SimpleNamespace(act=leaf))]))
    assert find_act_module(neox, 0) is leaf and find_act_module(gpt2, 0) is leaf
    with pytest.raises(ValueError, match="out of range"):
        find_act_module(neox, 1)
    with pytest.raises(ValueError, match="unsupported"):
        find_act_module(SimpleNamespace(), 0)


# ---- CLI ----

def test_cli_toy_build_writes_both_files(tmp_path, capsys):
    corpus = tmp_path / "c.txt"
    corpus.write_text("the cat sat\nthe dog ran\n", encoding="utf-8")
    out = tmp_path / "o"
    assert main(["build", "--model", "toy", "--layer", "0", "--corpus", str(corpus), "--out", str(out), "--top-k", "2"]) == 0
    a = json.loads((out / "atlas.json").read_text(encoding="utf-8"))
    assert a["n_texts"] == 2 and a["neurons_included"] > 0
    assert "<script" in (out / "atlas.html").read_text(encoding="utf-8")
    assert "wrote" in capsys.readouterr().out


def test_cli_html_rerender(tmp_path):
    src = tmp_path / "a.json"
    src.write_text(json.dumps(atlas()), encoding="utf-8")
    assert main(["html", str(src), "-o", str(tmp_path / "x.html")]) == 0
    assert (tmp_path / "x.html").read_text(encoding="utf-8") == render_html(atlas())


def test_cli_missing_corpus_fails(tmp_path, capsys):
    assert main(["build", "--model", "toy", "--corpus", str(tmp_path / "nope.txt")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_empty_corpus_fails(tmp_path, capsys):
    f = tmp_path / "e.txt"
    f.write_text("\n\n", encoding="utf-8")
    assert main(["build", "--model", "toy", "--corpus", str(f), "--out", str(tmp_path / "o")]) == 2
    assert "empty" in capsys.readouterr().err


def test_cli_html_rejects_non_atlas(tmp_path, capsys):
    f = tmp_path / "x.json"
    f.write_text("[1]", encoding="utf-8")
    assert main(["html", str(f)]) == 2


def test_module_entrypoint(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "neuron_atlas", "build", "--model", "toy", "--corpus", str(corpus), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert r.returncode == 0 and (tmp_path / "o" / "atlas.json").exists()


def test_saved_sample_matches_schema():
    a = json.loads((ROOT / "samples" / "atlas.json").read_text(encoding="utf-8"))
    assert a["model"] == "EleutherAI/pythia-14m" and a["n_neurons"] == 512 and len(a["neurons"]) == 512
    assert all(len(x["examples"]) == a["top_k"] for x in a["neurons"])
    assert (ROOT / "samples" / "atlas.html").read_text(encoding="utf-8") == render_html(a)


# ---- optional real model (needs torch + transformers + the model download) ----

@pytest.mark.skipif(os.environ.get("NEURON_ATLAS_REAL") != "1", reason="set NEURON_ATLAS_REAL=1 to run the real pythia-14m test")
def test_real_pythia_14m_matches_saved_sample():
    from neuron_atlas.providers import HFProvider
    texts = read_corpus(ROOT / "samples" / "corpus.txt")
    a = build_atlas(HFProvider("EleutherAI/pythia-14m"), texts, layer=3, top_k=5, skip_first=True, model_name="EleutherAI/pythia-14m")
    saved = json.loads((ROOT / "samples" / "atlas.json").read_text(encoding="utf-8"))
    assert a["n_neurons"] == 512 and a["n_tokens"] == saved["n_tokens"]
    assert n(a, 417)["top_tokens"][0]["token"] == n(saved, 417)["top_tokens"][0]["token"]
    assert abs(n(a, 417)["max_act"] - n(saved, 417)["max_act"]) < 0.05
