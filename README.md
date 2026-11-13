# neuron-atlas

Build a searchable atlas of the MLP neurons of a small GPT-2-class model, on CPU, with numpy only.

For every neuron you pick it records

- the **top-k activating tokens** with a context window (default k = 10, 8 tokens each side),
- **activation statistics**: mean, std, min, max, active fraction and sparsity,
- a **grouping of those top tokens** (same token, lower-cased, counted together),

over a text corpus (a small bundled one by default, or your own file), and writes one **self-contained HTML file** (search, sort, filter by layer, click a row for the examples; no network at view time) and one **JSON export**.

It shows what the numbers are. It does not label neurons and does not claim any neuron "means" anything.

## Quickstart (uses the saved sample, no model download, no numpy)

```bash
git clone https://github.com/MegaStretch23/neuron-atlas && cd neuron-atlas

# 1. open the saved atlas in a browser (single file, works offline)
xdg-open samples/atlas.html        # macOS: open ... / Windows: start ...

# 2. or search the saved JSON from the terminal (standard library only)
python3 neuron_atlas.py query samples/atlas.json --token the --limit 3 --examples 2
```

Output of step 2 on the saved sample (copied from a real run):

```
L4.N1867  max=6.337 mean=-0.015 sparsity=0.972  top tokens: 'the'x6, 'a'x2, 'on'x2
      6.337  ' not know who lives in the house on[ the] hill, but the lights are always on'
      6.086  ' allez-vous? The café on[ the] corner serves a very good croissant'
L0.N2158  max=5.946 mean=-0.047 sparsity=0.998  top tokens: 'greater'x2, 'arrived'x2, 'there'x1, 'supply'x1, 'large'x1
      5.946  'Once upon a time[ there] was a small fox who was afraid of'
      0.769  ". It produces most of the cell's[ supply] of adenosine triphosphate"
L5.N1660  max=5.905 mean=0.165 sparsity=0.890  top tokens: 'who'x2, 'there'x1, 'costs'x1, 'third'x1, 'spanish'x1
      5.905  'Once upon a time[ there] was a small fox who was afraid of'
      5.640  '.9 million a year earlier. Operating[ costs] rose 5% to $3.1'
```

One more, a neuron with sparsity 0.000 in this corpus (`--neuron L4.N840 --examples 2`):

```
L4.N840  max=8.733 mean=4.479 sparsity=0.000  top tokens: '<whitespace>'x2, 'def'x2, 'a'x2, 'x'x1, 're'x1
      8.733  '__ == "__main__":\n[ ]   data = load(sys.'
      8.664  '[def] load(path):\n   '
```

`samples/atlas.html` and `samples/atlas.json` are the real output of the command in the next section.

## Rebuild it yourself

```bash
python3 -m pip install numpy
python3 neuron_atlas.py fetch --model distilgpt2 --dest ./distilgpt2      # config, vocab, merges, model.safetensors (~353 MB)
python3 neuron_atlas.py build --model-dir ./distilgpt2 --model-name distilgpt2 \
        --layers 0-5 --top-neurons 100 --out out/
# -> out/atlas.html, out/atlas.json
```

Options: `--layers all|5|4,5|0-5`, `--top-k`, `--context`, `--top-neurons N` (neurons kept per layer, ranked by max activation; 0 = every neuron), `--neurons 5.12,5.77` (explicit list), `--corpus file.txt` (documents separated by blank lines), `--seq-len` (default 128), `--active-threshold` (default 0.5). `python3 neuron_atlas.py corpus` prints the bundled corpus. `neuron_atlas.py` is a single file with a PEP 723 header, so `uv run neuron_atlas.py ...` also works.

## What was measured (the saved sample)

| item | value |
|---|---|
| model | `distilgpt2` (Apache-2.0 per its model card), 6 layers, 12 heads, 768-d, 3072 MLP neurons per layer, weights sha256 `e1ff18884359fe8b...` (full hash in `samples/atlas.json`) |
| corpus | bundled: 49 short original passages, 2,504 tokens (9.8 KB) |
| selection | layers 0-5, the 100 neurons per layer with the highest max activation = 600 of 18,432 neurons |
| runtime | **5.24 s** total, 4.47 s of it the forward pass over 49 chunks (numpy 2.5.4, Python 3.13, 8 logical CPUs, AMD EPYC-Rome, x86_64, no GPU) |
| sizes | `atlas.html` 1.0 MB, `atlas.json` 1.5 MB |
| activation | post-GELU MLP hidden unit (the input of `mlp.c_proj`); the BOS position that starts each chunk is dropped |
| active / sparsity | active = activation > 0.5; sparsity = 1 - active fraction (change with `--active-threshold`) |

**Correctness check against transformers** (`samples/validate_against_transformers.py`, result in `samples/validation.json`): on the same 49 passages (2,553 positions including BOS), the numpy forward pass differs from `transformers` 5.22.2 / torch 2.15.0 (CPU) by at most **1.9e-4** in logits and **5.3e-5** in MLP activations, and the tokenizer produced identical token ids on all 49 passages. That is for distilgpt2 only. Other GPT-2-style checkpoints are untested.

Small things to know:

- The corpus is tiny, so each neuron's top-10 is the top-10 of about 2.5 thousand tokens. Use `--corpus` with more text for sturdier rankings. Neurons are picked by max activation, so the sample is the 600 most extreme ones, not a random sample. Across all 3,072 neurons per layer in this corpus, between 14 (layer 4) and 170 (layer 0) never exceed 0.5 (see `layer_summary` in the JSON).
- The GPT-2 regex uses `\p{L}` / `\p{N}`, which the standard library `re` lacks; the pre-tokenizer here uses `[^\W\d_]` and `\d`. Identical on the bundled corpus (checked above), possibly different on exotic Unicode digits.
- Supported: `gelu_new` / `gelu_fast` / `gelu_pytorch_tanh`, F32 / F16 / BF16 safetensors with or without the `transformer.` prefix. Anything else fails with a `FAILED: <cause>` line and exit code 1.
- The HTML viewer's JavaScript was syntax-checked with `node --check`; the tests check the embedded data and that the page has no external links or fetches. It has not been exercised in an automated browser run.
- Weights are not in this repo. You download them yourself from huggingface.co and the model's own license applies.

## Tests

```bash
python3 -m pip install numpy pytest
python3 -m pytest -q          # 35 tests, offline, about 0.3 s
```

The tests build a random-weight 2-layer model with a real byte-level BPE vocabulary in a temp folder, so CI needs no download. They check the forward pass against an independent loop-based implementation, causality, safetensors round-trips (F32/F16/BF16, truncated files), the tokenizer (merges, unicode round-trip), every statistic and top-k against brute force, chunk-boundary handling, determinism, HTML escaping of hostile corpus text, the CLI and its failure messages.

## Also by the author

Unrelated to this tool, there is no paid version of neuron-atlas. Other things I publish: a pay-per-event [SEC Form 4 insider cluster scanner](https://apify.com/johnveds15/sec-form4-insider-clusters) on Apify ($0.0004 per Form 4 filing parsed, $0.02 per insider cluster detected, price read from the Apify API on 2026-11-13), and two source-code packs for agent work on Gumroad: the [Agent-Turn Delta-Debugger CLI](https://vedovino.gumroad.com/l/zdwyic) ($25) and the [Agent log policy checker](https://vedovino.gumroad.com/l/agent-log-policy-checker) ($19), prices as listed on 2026-11-13.

## License

MIT, see `LICENSE`.
