import json, math, re, struct
from pathlib import Path

import numpy as np
import pytest

import neuron_atlas as na
from conftest import CFG, DOCS, make_model


# ---------------------------------------------------------------- tokenizer
def test_bytes_to_unicode_is_a_bijection_over_256_bytes():
    m = na.bytes_to_unicode()
    assert sorted(m) == list(range(256)) and len(set(m.values())) == 256


def test_split_pattern_matches_gpt2_pieces():
    assert na._PAT.findall("Hello, world! it's 2024_x") == ["Hello", ",", " world", "!", " it", "'s", " 2024", "_", "x"]
    assert na._PAT.findall("a  b\n\nc") == ["a", " ", " b", "\n", "\n", "c"]


def test_bpe_roundtrip_ascii_unicode_and_emoji(model_dir):
    bpe = na.BPE.from_dir(model_dir)
    for text in ["plain words here", "café naïve", "emoji \U0001f600 ok", "tabs\tand\nnewlines  ", ""]:
        assert bpe.decode_ids(bpe.encode(text)) == text


def test_bpe_applies_merges(model_dir):
    bpe = na.BPE.from_dir(model_dir)
    assert len(bpe.encode(" the")) == 1 and len(bpe.encode("in")) == 1
    assert bpe.inv[bpe.encode(" the")[0]] == "Ġthe"
    assert len(bpe.encode(" thx")) > 1


def test_bpe_unknown_symbol_is_a_clear_error(model_dir):
    bpe = na.BPE.from_dir(model_dir)
    del bpe.vocab["x"]
    with pytest.raises(na.AtlasError, match="not in vocab"):
        bpe.encode("x")


# ---------------------------------------------------------------- safetensors
@pytest.mark.parametrize("dtype,tol", [("F32", 0), ("F16", 1e-3), ("BF16", 1e-2)])
def test_safetensors_roundtrip(tmp_path, dtype, tol):
    t = {"a": np.random.default_rng(1).standard_normal((3, 5)).astype(np.float32), "b": np.arange(4, dtype=np.float32)}
    na.write_safetensors(tmp_path / "m.safetensors", t, dtype=dtype)
    got = na.read_safetensors(tmp_path / "m.safetensors")
    for k in t:
        assert got[k].shape == t[k].shape and got[k].dtype == np.float32
        assert np.abs(got[k] - t[k]).max() <= tol * (1 + np.abs(t[k]).max())


def test_safetensors_skips_unwanted_tensors(tmp_path):
    na.write_safetensors(tmp_path / "m.safetensors", {"keep": np.ones(2), "drop": np.ones(2)})
    assert list(na.read_safetensors(tmp_path / "m.safetensors", want=lambda n: n == "keep")) == ["keep"]


def test_safetensors_truncated_file_is_reported(tmp_path):
    na.write_safetensors(tmp_path / "m.safetensors", {"a": np.ones(100)})
    raw = (tmp_path / "m.safetensors").read_bytes()
    (tmp_path / "cut.safetensors").write_bytes(raw[:-50])
    with pytest.raises(na.AtlasError, match="outside the file"):
        na.read_safetensors(tmp_path / "cut.safetensors")
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(na.AtlasError, match="header length"):
        na.read_safetensors(tmp_path / "bad.safetensors")


# ---------------------------------------------------------------- numpy GPT-2
def test_gelu_new_known_values():
    x = np.array([0.0, 1.0, -10.0, 10.0], np.float32)
    y = na.gelu_new(x)
    assert y[0] == 0 and abs(y[1] - 0.8411920) < 1e-4 and abs(y[2]) < 1e-6 and abs(y[3] - 10) < 1e-4


def test_layer_norm_normalises():
    x = np.random.default_rng(0).standard_normal((5, 16)).astype(np.float32) * 7 + 3
    y = na.layer_norm(x, np.ones(16, np.float32), np.zeros(16, np.float32), 1e-5)
    assert np.abs(y.mean(-1)).max() < 1e-5 and np.abs(y.var(-1) - 1).max() < 1e-3


def _naive_forward(m: na.GPT2, ids):
    """Independent slow reference: python loops for attention, math.erf-free tanh gelu."""
    W, D, H = m.W, m.n_embd, m.n_head
    dh = D // H
    x = [W["wte.weight"][t] + W["wpe.weight"][p] for p, t in enumerate(ids)]
    ln = lambda v, w, b: (v - v.mean()) / math.sqrt(v.var() + m.eps) * w + b
    acts = {}
    for i in range(m.n_layer):
        h = [ln(v, W[f"h.{i}.ln_1.weight"], W[f"h.{i}.ln_1.bias"]) for v in x]
        qkv = [v @ W[f"h.{i}.attn.c_attn.weight"] + W[f"h.{i}.attn.c_attn.bias"] for v in h]
        new = []
        for t in range(len(ids)):
            heads = []
            for hd in range(H):
                q = qkv[t][hd * dh:(hd + 1) * dh]
                sc = np.array([q @ qkv[s][D + hd * dh:D + (hd + 1) * dh] / math.sqrt(dh) for s in range(t + 1)])
                p = np.exp(sc - sc.max()); p /= p.sum()
                heads.append(sum(p[s] * qkv[s][2 * D + hd * dh:2 * D + (hd + 1) * dh] for s in range(t + 1)))
            new.append(x[t] + np.concatenate(heads) @ W[f"h.{i}.attn.c_proj.weight"] + W[f"h.{i}.attn.c_proj.bias"])
        x = new
        a = []
        for t in range(len(ids)):
            pre = ln(x[t], W[f"h.{i}.ln_2.weight"], W[f"h.{i}.ln_2.bias"]) @ W[f"h.{i}.mlp.c_fc.weight"] + W[f"h.{i}.mlp.c_fc.bias"]
            a.append(0.5 * pre * (1 + np.tanh(math.sqrt(2 / math.pi) * (pre + 0.044715 * pre ** 3))))
        acts[i] = np.array(a)
        x = [x[t] + a[t] @ W[f"h.{i}.mlp.c_proj.weight"] + W[f"h.{i}.mlp.c_proj.bias"] for t in range(len(ids))]
    return acts


def test_forward_matches_naive_reference(model_dir):
    m = na.GPT2.load(model_dir)
    ids = [m.bos, 5, 77, 130, 3, 200, 9, 41]
    _, acts = m.run(ids, layers=[0, 1])
    ref = _naive_forward(m, ids)
    for l in (0, 1):
        assert acts[l].shape == (len(ids), 64) and np.abs(acts[l] - ref[l]).max() < 1e-4


def test_forward_is_causal(model_dir):
    m = na.GPT2.load(model_dir)
    a = [m.bos, 5, 77, 130, 3]
    b = a[:-1] + [99]
    _, x = m.run(a, layers=[1]); _, y = m.run(b, layers=[1])
    assert np.array_equal(x[1][:-1], y[1][:-1]) and not np.allclose(x[1][-1], y[1][-1])


def test_loader_handles_prefixless_f16_and_bf16(tmp_path):
    base = na.GPT2.load(make_model(tmp_path / "a"))
    for dtype in ("F16", "BF16"):
        m = na.GPT2.load(make_model(tmp_path / dtype, dtype=dtype, prefix=""))
        assert m.n_layer == base.n_layer and m.d_mlp == 64
        assert np.abs(m.logits([m.bos, 4, 5]) - base.logits([base.bos, 4, 5])).max() < 0.5


def test_unsupported_activation_and_missing_tensor(tmp_path):
    with pytest.raises(na.AtlasError, match="not supported"):
        na.GPT2.load(make_model(tmp_path / "x", cfg={**CFG, "activation_function": "relu"}))
    d = make_model(tmp_path / "y")
    na.write_safetensors(d / "model.safetensors", {"wte.weight": np.zeros((3, 3))})
    with pytest.raises(na.AtlasError, match="lacks tensors"):
        na.GPT2.load(d)
    with pytest.raises(na.AtlasError, match="not found"):
        na.GPT2.load(tmp_path / "nope")


def test_sequence_length_limits(model_dir):
    m = na.GPT2.load(model_dir)
    with pytest.raises(na.AtlasError):
        m.run([m.bos] * 65)
    with pytest.raises(na.AtlasError):
        m.run([])


# ---------------------------------------------------------------- atlas
def brute(model_dir, docs, layer, seq_len=16, thr=0.5):
    """Recompute activations chunk by chunk with the model directly; return (acts (n_tok, N), flat ids)."""
    m, bpe = na.GPT2.load(model_dir), na.BPE.from_dir(model_dir)
    rows, ids_all = [], []
    for d in docs:
        ids = bpe.encode(d)
        for s in range(0, len(ids), seq_len - 1):
            part = ids[s:s + seq_len - 1]
            _, a = m.run([m.bos] + part, layers=[layer])
            rows.append(a[layer][1:]); ids_all += part
    return np.concatenate(rows), ids_all


def test_stats_match_brute_force(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="1", top_k=3, top_neurons=0, seq_len=16, context=2)
    acts, ids = brute(model_dir, DOCS, 1)
    assert at["meta"]["corpus_tokens"] == len(ids) == acts.shape[0]           # BOS positions are not counted
    for n in at["neurons"]:
        col = acts[:, n["neuron"]]
        assert abs(n["max"] - col.max()) < 1e-3 and abs(n["min"] - col.min()) < 1e-3
        assert abs(n["mean"] - col.mean()) < 1e-4 and abs(n["std"] - col.std()) < 1e-4
        assert abs(n["active_frac"] - (col > 0.5).mean()) < 1e-4 and abs(n["sparsity"] - (1 - (col > 0.5).mean())) < 1e-4


def test_topk_equals_brute_force(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="0", top_k=4, top_neurons=0, seq_len=16, context=3)
    acts, ids = brute(model_dir, DOCS, 0)
    bpe = na.BPE.from_dir(model_dir)
    for n in at["neurons"]:
        want = np.sort(acts[:, n["neuron"]])[::-1][:4]
        got = [e["act"] for e in n["examples"]]
        assert np.allclose(got, want, atol=1e-3) and got == sorted(got, reverse=True)
        for e in n["examples"]:        # the reported token really sits at an index whose activation is that value
            hits = [g for g in range(len(ids)) if abs(acts[g, n["neuron"]] - e["act"]) < 1e-3 and bpe.decode_ids([ids[g]]) == e["token"]]
            assert hits


def test_fewer_tokens_than_k_gives_fewer_examples(model_dir):
    at = na.build_atlas(model_dir, ["h"], layers="0", top_k=10, top_neurons=2)
    assert all(len(n["examples"]) == 1 for n in at["neurons"])


def test_context_never_crosses_a_chunk_boundary(model_dir):
    doc = "the cat sat in the hat. " * 12
    at = na.build_atlas(model_dir, [doc], layers="0", top_k=5, top_neurons=0, seq_len=8, context=50)
    m = at["meta"]
    assert m["corpus_chunks"] > 3
    bpe = na.BPE.from_dir(model_dir)
    for n in at["neurons"][:20]:
        for e in n["examples"]:
            assert len(bpe.encode(e["before"] + e["token"] + e["after"])) <= 7     # seq_len-1 tokens per chunk


def test_example_text_is_consistent_with_the_corpus(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="1", top_k=3, top_neurons=5, context=4)
    for n in at["neurons"]:
        for e in n["examples"]:
            assert (e["before"] + e["token"] + e["after"]) in DOCS[e["doc"]]


def test_clusters_account_for_every_example(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="0-1", top_k=6, top_neurons=4)
    for n in at["neurons"]:
        assert sum(c["count"] for c in n["clusters"]) == len(n["examples"])
        assert [c["count"] for c in n["clusters"]] == sorted((c["count"] for c in n["clusters"]), reverse=True)


def test_top_neurons_picks_highest_max_and_explicit_list_wins(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="1", top_neurons=5, top_k=2)
    full = na.build_atlas(model_dir, DOCS, layers="1", top_neurons=0, top_k=2)
    best = sorted(full["neurons"], key=lambda n: (-n["max"], n["neuron"]))[:5]
    assert [n["neuron"] for n in at["neurons"]] == [n["neuron"] for n in best]
    ex = na.build_atlas(model_dir, DOCS, neurons=[(1, 7), (0, 3)], top_k=2)
    assert [(n["layer"], n["neuron"]) for n in ex["neurons"]] == [(0, 3), (1, 7)] and ex["meta"]["layers"] == [0, 1]


def test_argument_errors(model_dir):
    with pytest.raises(na.AtlasError, match="out of range"):
        na.build_atlas(model_dir, DOCS, layers="5")
    with pytest.raises(na.AtlasError, match="use 'all'"):
        na.build_atlas(model_dir, DOCS, layers="x")
    with pytest.raises(na.AtlasError, match="out of range"):
        na.build_atlas(model_dir, DOCS, neurons=[(0, 9999)])
    with pytest.raises(na.AtlasError, match="no tokens"):
        na.build_atlas(model_dir, [""], layers="0")
    with pytest.raises(na.AtlasError, match="top-k"):
        na.build_atlas(model_dir, DOCS, top_k=0)
    with pytest.raises(na.AtlasError, match="LAYER.NEURON"):
        na.parse_neurons("abc")
    assert na.parse_layers("0-1,1", 3) == [0, 1] and na.parse_neurons("L5.N12, 2.3") == [(5, 12), (2, 3)]


def test_atlas_is_deterministic(model_dir):
    a = na.build_atlas(model_dir, DOCS, top_k=3, top_neurons=6)
    b = na.build_atlas(model_dir, DOCS, top_k=3, top_neurons=6)
    a["meta"].pop("run"); b["meta"].pop("run")
    assert a == b


def test_meta_records_what_was_measured(model_dir):
    m = na.build_atlas(model_dir, DOCS, layers="0", top_neurons=1)["meta"]
    assert m["tool"] == "neuron-atlas" and m["mlp_neurons_per_layer"] == 64 and m["corpus_docs"] == 3
    assert len(m["model_sha256"]) == 64 and len(m["corpus_sha256"]) == 64 and m["run"]["seconds_total"] >= 0


# ---------------------------------------------------------------- html / json
def test_html_is_self_contained_and_data_roundtrips(model_dir):
    at = na.build_atlas(model_dir, DOCS, layers="0-1", top_k=3, top_neurons=4)
    html = na.render_html(at)
    assert not re.search(r"<link|<img|<iframe|\bsrc=|\bhref=|url\(|fetch\(|XMLHttpRequest|import\(|@import", html)
    blob = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert json.loads(blob) == json.loads(json.dumps(at))
    for needle in ('id="q"', 'id="layer"', "sort", "onclick"):
        assert needle in html


def test_html_escapes_hostile_corpus_text(model_dir):
    evil = ["x </script><script>alert(1)</script> & <b>y</b> the the the"]
    at = na.build_atlas(model_dir, evil, layers="0", top_k=3, top_neurons=3)
    html = na.render_html(at)
    assert html.count("</script>") == 2 and "<b>y" not in html and "alert(1)</script>" not in html
    assert json.loads(re.search(r'id="data" type="application/json">(.*?)</script>', html, re.S).group(1))["neurons"]


# ---------------------------------------------------------------- CLI
def test_cli_build_then_query(tmp_path, model_dir, capsys):
    corpus = tmp_path / "c.txt"
    corpus.write_text("the cat in the hat.\n\nthe end of the line here.\n")
    out = tmp_path / "out"
    assert na.main(["build", "--model-dir", str(model_dir), "--corpus", str(corpus), "--layers", "0-1", "--top-neurons", "5", "--top-k", "3", "--out", str(out)]) == 0
    assert (out / "atlas.html").stat().st_size > 1000 and json.loads((out / "atlas.json").read_text())["schema"] == na.SCHEMA
    assert "atlas: 10 neurons" in capsys.readouterr().out
    a = json.loads((out / "atlas.json").read_text())
    tok = a["neurons"][0]["clusters"][0]["token"]
    assert na.main(["query", str(out / "atlas.json"), "--token", tok, "--examples", "1", "--limit", "2"]) == 0
    assert "top tokens" in capsys.readouterr().out
    n0 = a["neurons"][0]
    assert na.main(["query", str(out / "atlas.json"), "--neuron", f"L{n0['layer']}.N{n0['neuron']}"]) == 0
    assert capsys.readouterr().out.count("max=") == 1


def test_cli_failures_say_FAILED_and_exit_1(tmp_path, model_dir, capsys):
    assert na.main(["build", "--model-dir", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 1
    assert "FAILED:" in capsys.readouterr().err
    empty = tmp_path / "e.txt"; empty.write_text("  \n")
    assert na.main(["build", "--model-dir", str(model_dir), "--corpus", str(empty), "--out", str(tmp_path / "o")]) == 1
    assert "has no text" in capsys.readouterr().err
    (tmp_path / "x.json").write_text('{"schema": "other"}')
    assert na.main(["query", str(tmp_path / "x.json")]) == 1
    assert na.main(["query", str(tmp_path / "nope.json")]) == 1


def test_query_filters_and_sorting():
    atlas = {"schema": na.SCHEMA, "neurons": [
        {"layer": 0, "neuron": 1, "max": 1, "mean": 0, "std": 0, "active_frac": .1, "sparsity": .9, "examples": [{"token": " The"}], "clusters": [{"token": "the", "count": 1}]},
        {"layer": 1, "neuron": 2, "max": 5, "mean": 0, "std": 0, "active_frac": .5, "sparsity": .5, "examples": [{"token": " then"}], "clusters": [{"token": "then", "count": 1}]}]}
    q = lambda **k: [(n["layer"], n["neuron"]) for n in na.query_atlas(atlas, **k)]
    assert q() == [(1, 2), (0, 1)] and q(token="the") == [(1, 2), (0, 1)] and q(token="the", exact=True) == [(0, 1)]
    assert q(layer=0) == [(0, 1)] and q(neuron="1.2") == [(1, 2)] and q(sort="sparsity_asc") == [(1, 2), (0, 1)] and q(limit=1) == [(1, 2)]


# ---------------------------------------------------------------- bundled corpus / fetch
def test_bundled_corpus_is_usable():
    docs = na.bundled_documents()
    assert len(docs) >= 40 and all(d.strip() for d in docs)
    assert sum(len(d) for d in docs) > 9000


def test_fetch_urls_name_the_four_files():
    assert [n for _, n in na.fetch_urls("distilgpt2")] == ["config.json", "vocab.json", "merges.txt", "model.safetensors"]
    assert all(u.startswith("https://huggingface.co/distilgpt2/resolve/main/") for u, _ in na.fetch_urls("distilgpt2"))


def test_saved_sample_matches_its_schema_and_html():
    s = Path(__file__).resolve().parent.parent / "samples"
    a = json.loads((s / "atlas.json").read_text(encoding="utf-8"))
    assert a["schema"] == na.SCHEMA and a["meta"]["neurons_in_atlas"] == len(a["neurons"]) > 0
    blob = re.search(r'id="data" type="application/json">(.*?)</script>', (s / "atlas.html").read_text(encoding="utf-8"), re.S).group(1)
    assert json.loads(blob)["neurons"] == a["neurons"]
