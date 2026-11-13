# neuron-atlas

Pick one layer of a tiny open language model, run a small text corpus through it on CPU, and see which tokens and contexts make each MLP neuron fire the most. The output is a static `atlas.json` and a single-file `atlas.html` you open in a browser. No server, no network once the model is downloaded.

It is a small exploration tool: it ranks the highest activations it saw in *your* corpus. It does not say what a neuron "means", and a short corpus shows only a slice of what a neuron does.

## Quickstart

```
pip install "neuron-atlas[model] @ git+https://github.com/MegaStretch23/neuron-atlas"
neuron-atlas build --model EleutherAI/pythia-14m --layer 3 --corpus samples/corpus.txt --out atlas-out --skip-first --top-k 5
```

(or clone the repo and `pip install -e ".[model]"`). `[model]` pulls `torch` and `transformers`; the ranking, JSON and HTML code is standard library only. The first run downloads the model from Hugging Face (pythia-14m is about 30 MB).

Saved run (the files are in `samples/`; CPU, 45 short sentences):

```
$ neuron-atlas build --model EleutherAI/pythia-14m --layer 3 --corpus samples/corpus.txt --out samples --skip-first --top-k 5
model=EleutherAI/pythia-14m layer=3 neurons=512/512 texts=45 tokens=704
wrote samples/atlas.json and samples/atlas.html
```

Open `samples/atlas.html` to browse the result: filter by token, sort by max or mean activation. Neuron 417 from that run, as printed from `samples/atlas.json` (activation, then the context with the firing token in brackets):

```
neuron 417  max 2.3926  mean -0.0379  top tokens: ' and' x5
   2.39   wrote Hamlet, Macbeth[ and] Othello, among many
   2.32   river flows through Brazil, Peru[ and] Colombia before reaching the Atlantic Ocean
   2.24  Tokyo, Osaka[ and] Kyoto are three of the largest
   2.03   faster chip, a better camera[ and] a larger battery.
   1.97  if x > 10[ and] y < 5: raise Value
```

In this corpus that neuron's five strongest contexts are all the word " and". Other neurons in the same file fire on mixed tokens; many top lists are not clean, which is typical for a tiny model and a 45-sentence corpus.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `EleutherAI/pythia-14m` | Pythia / GPT-NeoX or GPT-2 family model id from Hugging Face. `toy` = built-in no-download stand-in (hash buckets, not a language model) for trying the CLI |
| `--layer` | 3 | which transformer block's MLP to read |
| `--corpus` | required | text file, one text per line |
| `--top-k` | 10 | contexts kept per neuron |
| `--context` | 6 | tokens shown on each side of the firing token |
| `--max-length` | 128 | truncate each text to this many tokens |
| `--neurons` | all | comma-separated neuron ids |
| `--skip-first` | off | ignore the first token of each text |

`neuron-atlas html atlas.json -o atlas.html` re-renders the page from a saved JSON.

What is measured: the output of the MLP nonlinearity (the activation function inside the block, one value per hidden neuron per token), captured with a forward hook. Rankings are over every token of every text, ties broken by corpus order. `atlas.html` draws the first 300 matching neurons at a time.

Library use: `build_atlas(provider, texts, layer)` takes any object with `run(text, layer) -> (tokens, rows)`, so you can rank activations from your own model.

## Tests

```
pip install pytest && pytest
```

39 tests run without a download, using a fixed-activation fake model: ranking order, ties, context edges, skip-first, neuron subsets, bad input, JSON/HTML output (no external requests, script-breaking text escaped), CLI exit codes. One more test runs the real model and compares with the saved sample: `NEURON_ATLAS_REAL=1 pytest` (needs `torch`, `transformers` and the download).

## Hosted / paid options

neuron-atlas is free and has no paid version. Other things I sell (prices read from the Apify and Gumroad APIs on 2026-11-13):

- Apify, pay per event: [AI API Key Leak Scanner](https://apify.com/johnveds15/ai-api-key-leak-scanner?utm_source=github&utm_medium=readme&utm_campaign=neuron-atlas) ($0.01 per repository or URL scanned), [SEC Form 4 insider cluster scanner](https://apify.com/johnveds15/sec-form4-insider-clusters?utm_source=github&utm_medium=readme&utm_campaign=neuron-atlas) ($0.0004 per Form 4 filing parsed, $0.02 per insider cluster detected).
- Gumroad source packs: [Doom-detector for headless Claude Code runs](https://vedovino.gumroad.com/l/mjxcmw?utm_source=github&utm_medium=readme&utm_campaign=neuron-atlas) ($15), [pdfguard: find hidden prompt-injection text in PDFs](https://vedovino.gumroad.com/l/gvvcsh?utm_source=github&utm_medium=readme&utm_campaign=neuron-atlas) ($19).

MIT licensed.
