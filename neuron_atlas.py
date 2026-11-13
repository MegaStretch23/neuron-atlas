#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24"]
# ///
"""neuron-atlas: record what the MLP neurons of a small GPT-2-class model respond to, on CPU, with numpy only.

    python3 neuron_atlas.py fetch --model distilgpt2 --dest ./distilgpt2
    python3 neuron_atlas.py build --model-dir ./distilgpt2 --layers 0-5 --top-neurons 100 --out out/
    python3 neuron_atlas.py query samples/atlas.json --token the      # stdlib only, no model, no numpy
    python3 neuron_atlas.py corpus                                    # print the bundled corpus

For every selected MLP neuron (the post-GELU hidden unit of a transformer block) it records the top-k activating
tokens with a context window, activation statistics (mean, std, min, max, active fraction, sparsity) and a grouping of
those top tokens, over a text corpus (a small bundled one by default). Output: one self-contained HTML file
(searchable, sortable, no network at view time) and one JSON export.

The forward pass is a plain numpy GPT-2 (safetensors reader, byte-level BPE, causal attention, gelu_new). It reads the
HuggingFace layout (config.json, model.safetensors, vocab.json, merges.txt). The atlas shows measured activations; it
does not label neurons or claim they mean anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path

__version__ = "0.1.0"
SCHEMA = "neuron-atlas/1"


class AtlasError(Exception):
    """A readable failure (bad model dir, bad argument); the CLI prints it as FAILED: <cause> and exits 1."""


# ----------------------------------------------------------------------------- bundled corpus
# Original short passages written for this repo (no third-party text). Blank line = new document.
BUNDLED_CORPUS = """\
The committee met on Tuesday morning to review the budget for the coming year. After two hours of discussion, the members voted seven to two in favour of the revised plan, which cuts travel costs by twelve percent and adds funding for the public library.

To make the sauce, heat two tablespoons of olive oil in a large pan over medium heat. Add one chopped onion and cook for five minutes, stirring often, until soft. Stir in three cloves of garlic, a pinch of salt, and a can of crushed tomatoes. Simmer for twenty minutes.

def moving_average(values, window):
    total = 0.0
    out = []
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out

On 14 March 2019 the station recorded a temperature of 21.4 degrees, the warmest March day since 1968. By 6 p.m. the reading had fallen to 15 degrees, and rain was forecast for Thursday and Friday.

"Did you lock the door?" she asked.
"I think so," he said. "I mean, yes. I am almost sure."
"Almost is not a word I like at midnight," she said, and went back to check.

SELECT customers.name, COUNT(orders.id) AS order_count, SUM(orders.total) AS revenue FROM customers JOIN orders ON orders.customer_id = customers.id WHERE orders.created_at >= '2024-01-01' GROUP BY customers.name ORDER BY revenue DESC LIMIT 10;

A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. The first few primes are 2, 3, 5, 7, 11, 13, 17 and 19. Every integer greater than 1 can be written as a product of primes in exactly one way, apart from the order of the factors.

Dear Ms. Alvarez, thank you for your letter of 3 June. We are sorry to hear that the parcel arrived damaged. Please send us the order number and a photograph of the box, and we will arrange a replacement or a refund within five working days. Yours sincerely, Customer Care.

The mitochondrion is an organelle found in most eukaryotic cells. It produces most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. Mitochondria have their own small genome, inherited in most species from the mother.

git checkout -b fix/login-timeout
git add src/auth.py tests/test_auth.py
git commit -m "Raise session timeout to 30 minutes"
git push origin fix/login-timeout

Shopping list: eggs, milk, brown bread, two lemons, a bag of rice, coffee beans, dish soap, tomatoes, basil, and something sweet for Sunday.

The old lighthouse stood at the end of the cliff road, its white paint peeling in long strips. Every evening the keeper climbed the ninety-two steps, lit the lamp, and watched the ships crawl along the horizon like slow beads on a string.

function debounce(fn, delay) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

Revenue for the third quarter was $4.2 million, up 8% from $3.9 million a year earlier. Operating costs rose 5% to $3.1 million. The board approved a dividend of $0.15 per share, payable on 30 November to holders of record on 15 November.

Bonjour, comment allez-vous? The café on the corner serves a very good croissant, and the naïve tourist who asked for "a coffee, to go" was handed a paper cup and a smile.

WARNING: do not operate heavy machinery while taking this medication. Keep out of reach of children. Store below 25 C in a dry place. If symptoms persist for more than seven days, consult a doctor or pharmacist.

# Installation

1. Download the archive from the releases page.
2. Unpack it into any folder.
3. Run `./setup.sh` and follow the prompts.
4. Restart your terminal.

If you see the error "command not found", check that the folder is on your PATH.

Chess is played on a board of 64 squares. Each player starts with sixteen pieces: one king, one queen, two rooks, two bishops, two knights and eight pawns. White moves first. The game ends when a king is checkmated, when a player resigns, or when the position is drawn.

I do not know who lives in the house on the hill, but the lights are always on at three in the morning, and the dog never barks, which is stranger than the lights.

import json
import sys

def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)

if __name__ == "__main__":
    data = load(sys.argv[1])
    print(len(data), "records")

The river rises in the northern hills and flows south for about 340 kilometres before reaching the sea. In spring the melting snow doubles its volume; in late summer it shrinks to a string of pools between gravel banks.

Meeting notes, Monday: the release is moved to Friday. Priya will finish the migration script. Tom owns the dashboard fix. Open question: do we keep the old API for another month? Decision: yes, until 1 December, then remove it.

Photosynthesis converts light energy into chemical energy. In plants, the reaction takes place in the chloroplasts, where chlorophyll absorbs mostly red and blue light. Carbon dioxide and water go in; glucose and oxygen come out.

URL parameters are separated by ampersands, for example https://example.com/search?q=lamp&page=2&sort=price. The path is case sensitive on most servers, while the host name is not. An email address such as someone@example.org follows a different grammar.

The train to Edinburgh leaves from platform 4 at 08:15 and arrives at 12:47. Passengers with reserved seats should board coach C. A trolley service runs between London and York only. Please keep your ticket until you leave the station.

Q: How many days are in a leap year? A: 366. Q: Which planet is closest to the sun? A: Mercury. Q: What is the chemical symbol for gold? A: Au. Q: Who wrote Hamlet? A: William Shakespeare.

Once upon a time there was a small fox who was afraid of the dark. Every night she curled up in her den and counted the stars she could see through the hole in the roof. One, two, three, four, five. When she reached ten she was always asleep.

The patient is a 54-year-old man who presented with chest pain lasting twenty minutes. Blood pressure was 138 over 86, heart rate 92, oxygen saturation 97 percent on room air. An electrocardiogram showed no acute changes. He was discharged with follow-up in one week.

export PATH="$HOME/bin:$PATH"
for f in *.log; do
  gzip -9 "$f"
done
echo "compressed $(ls *.gz | wc -l) files"

Interest on the loan accrues daily at an annual rate of 6.5 percent. Late payments incur a fee of 25 dollars. The borrower may repay the principal in whole or in part at any time without penalty, by giving written notice not less than ten days in advance.

Wow, that was the loudest thunder I have heard all year! The windows rattled, the cat shot under the sofa, and for a second every light in the street went out. Then the rain came, hard and sudden, drumming on the roof.

The museum is open from Tuesday to Sunday, 10 a.m. to 5 p.m., and closed on Mondays and public holidays. Admission is free for children under twelve. Guided tours in English, German and Spanish start every hour on the hour.

In 1969 two astronauts walked on the surface of the Moon, while a third waited in orbit. The mission lasted just over eight days. The astronauts collected about twenty-one kilograms of lunar rock and soil and brought it back to Earth.

class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG. the quick brown fox jumps over the lazy dog. The Quick Brown Fox Jumps Over The Lazy Dog.

Her grandmother used to say that a house is not a home until someone has burned the toast in it. The kitchen smelled of coffee and smoke, the radio muttered about traffic, and somewhere upstairs a floorboard creaked in a familiar way.

Table 2 shows the results. Accuracy rose from 71.3 to 78.9 percent when the training set was doubled, while the standard deviation across five runs stayed near 0.4. We did not test larger sets, so we make no claim about them.

Hi team, quick update: the server was restarted at 02:10 UTC after the disk filled up. Logs older than 30 days were removed. No data was lost. We added an alert at 80 percent usage so we hear about it earlier next time. Thanks, Sam.

Tickets cost 12 euros for adults and 7 euros for students. Groups of ten or more get a discount of 15 percent. Payment is accepted in cash or by card, but not by cheque.

The sparrows arrived before dawn, dozens of them, and settled on the wire like notes on a staff. When the first bus rumbled past they scattered all at once, then drifted back, one by one, to the same places.

if (user.isLoggedIn() && !user.isBanned()) {
  renderDashboard(user.id);
} else {
  redirect("/login?next=" + encodeURIComponent(location.pathname));
}

A good password is long, unique and random. Eight characters is too short; sixteen or more is better. A password manager can generate and store them, so you only need to remember one strong passphrase, such as four unrelated words.

Section 4. Termination. Either party may terminate this agreement by giving thirty days' written notice to the other party. Upon termination, all outstanding invoices become due immediately, and each party shall return the other party's confidential materials.

x = 3, y = 4, so the distance from the origin is the square root of 3 squared plus 4 squared, which is 5. In general, a squared plus b squared equals c squared for the sides of a right triangle.

The market opens at nine. By ten the stalls are crowded: pears and plums, wool socks, brass keys, jars of honey, a man selling secondhand books by the kilo, and a girl who will tell your fortune for a coin.
"""


def bundled_documents() -> list[str]:
    return [d.strip("\n") for d in re.split(r"\n\s*\n(?=\S)", BUNDLED_CORPUS) if d.strip()]


# ----------------------------------------------------------------------------- safetensors (stdlib reader/writer)
_DTYPES = {"F32": "<f4", "F16": "<f2", "BF16": None, "F64": "<f8"}


def _np():
    import numpy as np
    return np


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise AtlasError(f"{path.name}: file too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or n > size - 8 or n > 100_000_000:
            raise AtlasError(f"{path.name}: bad safetensors header length {n}")
        try:
            header = json.loads(f.read(n))
        except ValueError as e:
            raise AtlasError(f"{path.name}: safetensors header is not JSON ({e})")
    return header, 8 + n


def read_safetensors(path: str | Path, want=None) -> dict:
    """Return {name: float32 ndarray}. `want` = optional predicate on the tensor name (others are skipped, never read)."""
    np = _np()
    path = Path(path)
    header, base = read_safetensors_header(path)
    size = path.stat().st_size
    out = {}
    with open(path, "rb") as f:
        for name, meta in header.items():
            if name == "__metadata__" or (want is not None and not want(name)):
                continue
            dtype, shape, (a, b) = meta["dtype"], meta["shape"], meta["data_offsets"]
            if dtype not in _DTYPES:
                raise AtlasError(f"{name}: unsupported dtype {dtype}")
            if base + b > size or a > b:
                raise AtlasError(f"{name}: data offsets {a}:{b} outside the file (truncated download?)")
            f.seek(base + a)
            buf = f.read(b - a)
            if dtype == "BF16":
                arr = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                arr = np.frombuffer(buf, dtype=_DTYPES[dtype]).astype(np.float32)
            if arr.size != int(np.prod(shape, dtype=np.int64)):
                raise AtlasError(f"{name}: {arr.size} values for shape {shape}")
            out[name] = arr.reshape(shape)
    return out


def write_safetensors(path: str | Path, tensors: dict, dtype: str = "F32") -> None:
    """Minimal writer (used by the tests to build a random-weight model; also handy for converting)."""
    np = _np()
    header, chunks, pos = {"__metadata__": {"format": "pt"}}, [], 0
    for name, t in tensors.items():
        t = np.ascontiguousarray(t, dtype=np.float32)
        if dtype == "F32":
            buf = t.astype("<f4").tobytes()
        elif dtype == "F16":
            buf = t.astype("<f2").tobytes()
        elif dtype == "BF16":
            buf = (t.view(np.uint32) >> 16).astype("<u2").tobytes()
        else:
            raise AtlasError(f"cannot write dtype {dtype}")
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [pos, pos + len(buf)]}
        chunks.append(buf)
        pos += len(buf)
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(chunks))


# ----------------------------------------------------------------------------- GPT-2 byte-level BPE
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


# GPT-2's split pattern uses \p{L} / \p{N}; the stdlib `re` has no \p{..}, so letters = [^\W\d_], numbers = \d.
# Identical on ASCII text; may differ on rare Unicode categories (other numeric symbols).
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


class BPE:
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.inv = {i: t for t, i in vocab.items()}
        self.ranks = {p: i for i, p in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {u: b for b, u in self.b2u.items()}
        self._cache: dict[str, list[str]] = {}

    @classmethod
    def from_dir(cls, d: str | Path) -> "BPE":
        d = Path(d)
        try:
            vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
            lines = (d / "merges.txt").read_text(encoding="utf-8").split("\n")
        except (OSError, ValueError) as e:
            raise AtlasError(f"cannot read tokenizer files in {d}: {e}")
        merges = [tuple(l.split(" ")) for l in lines if l and not l.startswith("#version")]
        if any(len(m) != 2 for m in merges):
            raise AtlasError("merges.txt has a line that is not 'a b'")
        return cls(vocab, merges)

    def _bpe(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, 1 << 60))
            if best not in self.ranks:
                break
            a, b = best
            new, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        self._cache[token] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids = []
        for piece in _PAT.findall(text):
            mapped = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for sym in self._bpe(mapped):
                if sym not in self.vocab:
                    raise AtlasError(f"token {sym!r} is not in vocab.json")
                ids.append(self.vocab[sym])
        return ids

    def token_bytes(self, tid: int) -> bytes:
        return bytes(self.u2b[c] for c in self.inv[tid])

    def decode_ids(self, ids) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------- numpy GPT-2
def gelu_new(x):
    np = _np()
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def layer_norm(x, w, b, eps):
    np = _np()
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


_ACTS = {"gelu_new": gelu_new, "gelu_fast": gelu_new, "gelu_pytorch_tanh": gelu_new}


class GPT2:
    def __init__(self, cfg: dict, W: dict):
        self.cfg = cfg
        self.n_layer = int(cfg["n_layer"])
        self.n_head = int(cfg["n_head"])
        self.n_embd = int(cfg["n_embd"])
        self.n_pos = int(cfg.get("n_positions", cfg.get("n_ctx", 1024)))
        self.eps = float(cfg.get("layer_norm_epsilon", 1e-5))
        act = cfg.get("activation_function", "gelu_new")
        if act not in _ACTS:
            raise AtlasError(f"activation_function {act!r} is not supported (supported: {', '.join(sorted(_ACTS))})")
        self.act = _ACTS[act]
        if self.n_embd % self.n_head:
            raise AtlasError("n_embd is not divisible by n_head")
        self.W = W
        self.bos = int(cfg.get("bos_token_id", 0))
        self.d_mlp = W["h.0.mlp.c_fc.weight"].shape[1] if self.n_layer else 0

    @classmethod
    def load(cls, model_dir: str | Path) -> "GPT2":
        d = Path(model_dir)
        if not d.is_dir():
            raise AtlasError(f"model directory {d} not found (run: neuron_atlas.py fetch --model distilgpt2 --dest {d})")
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise AtlasError(f"cannot read config.json in {d}: {e}")
        st = d / "model.safetensors"
        if not st.exists():
            raise AtlasError(f"{st} not found (run: neuron_atlas.py fetch --model distilgpt2 --dest {d})")
        raw = read_safetensors(st, want=lambda n: not n.endswith(".attn.bias") and not n.endswith(".masked_bias"))
        W = {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in raw.items()}
        need = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
        for i in range(int(cfg["n_layer"])):
            need += [f"h.{i}.{p}" for p in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight",
                                           "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
                                           "mlp.c_proj.weight", "mlp.c_proj.bias")]
        missing = [n for n in need if n not in W]
        if missing:
            raise AtlasError(f"model.safetensors lacks tensors: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
        return cls(cfg, W)

    def _attn(self, x, i):
        np = _np()
        W, T, D, H = self.W, x.shape[0], self.n_embd, self.n_head
        qkv = x @ W[f"h.{i}.attn.c_attn.weight"] + W[f"h.{i}.attn.c_attn.bias"]
        q, k, v = (qkv[:, j * D:(j + 1) * D].reshape(T, H, D // H).transpose(1, 0, 2) for j in range(3))
        s = (q @ k.transpose(0, 2, 1)) / math.sqrt(D // H)
        s = np.where(np.tril(np.ones((T, T), dtype=bool)), s, np.float32(-1e9))
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        o = (p @ v).transpose(1, 0, 2).reshape(T, D)
        return o @ W[f"h.{i}.attn.c_proj.weight"] + W[f"h.{i}.attn.c_proj.bias"]

    def run(self, ids, layers=()):
        """Forward `ids`. Returns (hidden_after_last_run_block, {layer: post-GELU MLP activations (T, d_mlp)}). Stops after max(layers)."""
        np = _np()
        T = len(ids)
        if T == 0 or T > self.n_pos:
            raise AtlasError(f"sequence length {T} outside 1..{self.n_pos}")
        W = self.W
        x = W["wte.weight"][np.asarray(ids)] + W["wpe.weight"][:T]
        acts = {}
        last = max(layers) if layers else self.n_layer - 1
        for i in range(last + 1):
            x = x + self._attn(layer_norm(x, W[f"h.{i}.ln_1.weight"], W[f"h.{i}.ln_1.bias"], self.eps), i)
            h = self.act(layer_norm(x, W[f"h.{i}.ln_2.weight"], W[f"h.{i}.ln_2.bias"], self.eps) @ W[f"h.{i}.mlp.c_fc.weight"] + W[f"h.{i}.mlp.c_fc.bias"])
            if i in layers:
                acts[i] = h
            x = x + h @ W[f"h.{i}.mlp.c_proj.weight"] + W[f"h.{i}.mlp.c_proj.bias"]
        return x, acts

    def logits(self, ids):
        x, _ = self.run(ids)
        return layer_norm(x, self.W["ln_f.weight"], self.W["ln_f.bias"], self.eps) @ self.W["wte.weight"].T


# ----------------------------------------------------------------------------- atlas builder
def parse_layers(spec: str, n_layer: int) -> list[int]:
    if spec in ("all", "*"):
        return list(range(n_layer))
    out = set()
    try:
        for part in spec.split(","):
            if "-" in part:
                a, b = part.split("-")
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    except ValueError:
        raise AtlasError(f"--layers {spec!r}: use 'all', '5', '4,5' or '0-5'")
    bad = [l for l in out if not 0 <= l < n_layer]
    if bad or not out:
        raise AtlasError(f"--layers {spec!r}: layer out of range 0..{n_layer - 1}")
    return sorted(out)


def parse_neurons(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        m = re.fullmatch(r"\s*L?(\d+)\.N?(\d+)\s*", part)
        if not m:
            raise AtlasError(f"neuron {part!r}: use LAYER.NEURON, e.g. 5.123 or L5.N123")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def load_corpus(path: str | None) -> list[str]:
    if not path:
        return bundled_documents()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise AtlasError(f"cannot read corpus {path}: {e}")
    docs = [d.strip("\n") for d in re.split(r"\n\s*\n(?=\S)", text) if d.strip()]
    if not docs:
        raise AtlasError(f"corpus {path} has no text")
    return docs


def _cpu_model() -> str | None:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build_atlas(model_dir, docs, layers="all", top_k=10, context=8, top_neurons=100, neurons=None, seq_len=128,
                active_threshold=0.5, model_name=None, corpus_name="bundled") -> dict:
    np = _np()
    t0 = time.perf_counter()
    if top_k < 1 or context < 0 or seq_len < 2 or top_neurons < 0:
        raise AtlasError("need top-k >= 1, context >= 0, seq-len >= 2, top-neurons >= 0")
    model = GPT2.load(model_dir)
    bpe = BPE.from_dir(model_dir)
    seq_len = min(seq_len, model.n_pos)
    sel_layers = parse_layers(layers, model.n_layer) if isinstance(layers, str) else sorted(layers)
    if neurons:
        for l, n in neurons:
            if l not in range(model.n_layer) or n not in range(model.d_mlp):
                raise AtlasError(f"neuron {l}.{n} out of range (layers 0..{model.n_layer - 1}, neurons 0..{model.d_mlp - 1})")
        sel_layers = sorted({l for l, _ in neurons})
    N = model.d_mlp
    bos = model.bos
    if bos not in bpe.inv:
        raise AtlasError(f"bos_token_id {bos} is not in the vocabulary")

    # tokenize; chunk each doc into seq_len-1 tokens (+ BOS in front, whose activations are discarded)
    flat_ids, flat_doc, chunks = [], [], []   # chunks: (start, end) in flat arrays
    for di, doc in enumerate(docs):
        ids = bpe.encode(doc)
        for s in range(0, len(ids), seq_len - 1):
            part = ids[s:s + seq_len - 1]
            chunks.append((len(flat_ids), len(flat_ids) + len(part)))
            flat_ids.extend(part)
            flat_doc.extend([di] * len(part))
    if not flat_ids:
        raise AtlasError("the corpus produced no tokens")
    flat_ids_a = np.asarray(flat_ids)
    chunk_starts = np.asarray([c[0] for c in chunks])

    L = {l: {"v": np.full((N, top_k), -np.inf, np.float32), "i": np.full((N, top_k), -1, np.int64),
             "sum": np.zeros(N), "sq": np.zeros(N), "act": np.zeros(N, np.int64),
             "max": np.full(N, -np.inf, np.float32), "min": np.full(N, np.inf, np.float32)} for l in sel_layers}
    tf = time.perf_counter()
    for (a, b) in chunks:
        _, acts = model.run([bos] + flat_ids[a:b], layers=sel_layers)
        gidx = np.arange(a, b)
        for l in sel_layers:
            x = acts[l][1:]                                        # drop BOS position -> (T, N)
            S = L[l]
            S["sum"] += x.sum(0, dtype=np.float64)
            S["sq"] += (x.astype(np.float64) ** 2).sum(0)
            S["act"] += (x > active_threshold).sum(0)
            S["max"] = np.maximum(S["max"], x.max(0))
            S["min"] = np.minimum(S["min"], x.min(0))
            cv = np.concatenate([S["v"], x.T], axis=1)
            ci = np.concatenate([S["i"], np.broadcast_to(gidx, (N, len(gidx)))], axis=1)
            part = np.argpartition(-cv, top_k - 1, axis=1)[:, :top_k]
            S["v"], S["i"] = np.take_along_axis(cv, part, 1), np.take_along_axis(ci, part, 1)
    t_forward = time.perf_counter() - tf
    n_tok = len(flat_ids)

    def text(a, b):
        return bpe.decode_ids(flat_ids[a:b]) if b > a else ""

    # choose neurons
    chosen = []
    if neurons:
        chosen = sorted(set(neurons))
    else:
        for l in sel_layers:
            order = np.lexsort((np.arange(N), -L[l]["max"]))
            take = order if top_neurons == 0 else order[:top_neurons]
            chosen += [(l, int(n)) for n in take]
    out_neurons = []
    for l, n in chosen:
        S = L[l]
        o = np.argsort(-S["v"][n], kind="stable")
        examples = []
        for j in o:
            g = int(S["i"][n, j])
            if g < 0:
                continue
            ci = int(np.searchsorted(chunk_starts, g, side="right") - 1)
            cs, ce = chunks[ci]
            examples.append({"doc": flat_doc[g], "before": text(max(cs, g - context), g), "token": text(g, g + 1),
                             "after": text(g + 1, min(ce, g + 1 + context)), "act": round(float(S["v"][n, j]), 4)})
        groups: dict[str, list[float]] = {}
        for e in examples:
            groups.setdefault(e["token"].strip().lower() or "<whitespace>", []).append(e["act"])
        clusters = sorted(({"token": k, "count": len(v), "max": max(v), "mean": round(sum(v) / len(v), 4)} for k, v in groups.items()),
                          key=lambda c: (-c["count"], -c["max"], c["token"]))
        mean = S["sum"][n] / n_tok
        var = max(S["sq"][n] / n_tok - mean * mean, 0.0)
        active = float(S["act"][n] / n_tok)
        out_neurons.append({"layer": l, "neuron": n, "max": round(float(S["max"][n]), 4), "min": round(float(S["min"][n]), 4),
                            "mean": round(float(mean), 5), "std": round(math.sqrt(var), 5), "active_frac": round(active, 5),
                            "sparsity": round(1 - active, 5), "examples": examples, "clusters": clusters})
    layer_summary = [{"layer": l, "neurons": N, "mean_active_frac": round(float(L[l]["act"].mean() / n_tok), 5),
                      "never_active_in_corpus": int((L[l]["act"] == 0).sum())} for l in sel_layers]
    return {"schema": SCHEMA,
            "meta": {"tool": "neuron-atlas", "version": __version__, "model": model_name or Path(model_dir).resolve().name,
                     "model_sha256": _sha256(Path(model_dir) / "model.safetensors"),
                     "n_layer": model.n_layer, "n_head": model.n_head, "n_embd": model.n_embd, "mlp_neurons_per_layer": N,
                     "layers": sel_layers, "top_k": top_k, "context_tokens": context, "seq_len": seq_len, "active_threshold": active_threshold,
                     "corpus": corpus_name, "corpus_docs": len(docs), "corpus_tokens": n_tok, "corpus_chunks": len(chunks),
                     "corpus_sha256": hashlib.sha256("\n\n".join(docs).encode("utf-8")).hexdigest(),
                     "neurons_in_atlas": len(out_neurons), "neuron_selection": "explicit" if neurons else (f"top {top_neurons} per layer by max activation" if top_neurons else "all"),
                     "activation": "post-GELU MLP hidden unit (input of mlp.c_proj); first (BOS) position of each chunk excluded",
                     "sparsity_definition": "1 - fraction of corpus tokens with activation > active_threshold",
                     "run": {"seconds_total": round(time.perf_counter() - t0, 2), "seconds_forward": round(t_forward, 2),
                             "python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
                             "cpu_model": _cpu_model(), "cpu_count": os.cpu_count()}},
            "layer_summary": layer_summary, "neurons": out_neurons}


# ----------------------------------------------------------------------------- HTML
_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>neuron-atlas</title>
<style>
body{font:14px/1.45 system-ui,sans-serif;margin:16px;color:#1b1b1b;background:#fafafa}
h1{font-size:20px;margin:0 0 4px}.sub{color:#555;margin-bottom:10px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0}
input,select,button{font:inherit;padding:4px 6px}input{min-width:260px}
table{border-collapse:collapse;width:100%;background:#fff}th,td{border-bottom:1px solid #e3e3e3;padding:4px 8px;text-align:left;vertical-align:top}
th{cursor:pointer;background:#f0f0f0;position:sticky;top:0}th.on::after{content:" \25BE"}th.on.asc::after{content:" \25B4"}
tr.row{cursor:pointer}tr.row:hover{background:#f6f9ff}td.num{font-variant-numeric:tabular-nums;text-align:right}
.ex{font-family:ui-monospace,Menlo,monospace;white-space:pre-wrap;margin:2px 0}.ex mark{background:#ffd54f;padding:0 1px}.ex .a{color:#777}
.cl{color:#333}.cl b{font-weight:600}.small{color:#666;font-size:12px}
</style></head><body>
<h1>neuron-atlas</h1><div class="sub" id="meta"></div>
<div class="controls"><input id="q" type="search" placeholder="search a token (e.g. the) or a neuron (L5.N123)" aria-label="search">
<select id="layer" aria-label="layer"></select><button id="reset" type="button">reset</button><span class="small" id="count"></span></div>
<table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table>
<p class="small">Activations are the post-GELU MLP hidden units measured on the corpus named above; click a row for its top-activating tokens. Nothing here labels what a neuron "means".</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById("data").textContent),M=D.meta,ns=D.neurons;
var cols=[["layer","layer"],["neuron","neuron"],["max","max"],["mean","mean"],["std","std"],["active_frac","active frac"],["sparsity","sparsity"],["top","top tokens (count)"]];
var st={key:"max",asc:false},body=document.getElementById("body"),head=document.getElementById("head");
function el(t,c,x){var e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
function vis(s){return s.replace(/\n/g,"↵").replace(/\t/g,"→")}
document.getElementById("meta").textContent=M.model+" · layers "+M.layers.join(",")+" · "+M.neurons_in_atlas+" neurons ("+M.neuron_selection+") · corpus: "+M.corpus+", "+M.corpus_docs+" docs, "+M.corpus_tokens+" tokens · top-"+M.top_k+" examples · active = activation > "+M.active_threshold;
var sel=document.getElementById("layer");var o=el("option",null,"all layers");o.value="";sel.appendChild(o);
M.layers.forEach(function(l){var o=el("option",null,"layer "+l);o.value=String(l);sel.appendChild(o)});
cols.forEach(function(c){var th=el("th",null,c[1]);th.dataset.k=c[0];th.onclick=function(){if(c[0]==="top")return;if(st.key===c[0])st.asc=!st.asc;else{st.key=c[0];st.asc=(c[0]==="layer"||c[0]==="neuron")}render()};head.appendChild(th)});
function topTok(n){return n.clusters.slice(0,4).map(function(c){return c.token+" ("+c.count+")"}).join(", ")}
function match(n,q){if(!q)return true;var m=/^l?(\d+)\.n?(\d+)$/i.exec(q);if(m)return n.layer===+m[1]&&n.neuron===+m[2];
 var s=q.toLowerCase();return n.clusters.some(function(c){return c.token.indexOf(s)>=0})||n.examples.some(function(e){return e.token.toLowerCase().indexOf(s)>=0})}
function example(e){var d=el("div","ex");d.appendChild(el("span","a",vis(e.before)));d.appendChild(el("mark",null,vis(e.token)));d.appendChild(el("span","a",vis(e.after)));d.appendChild(el("span","small","  "+e.act.toFixed(3)+" (doc "+e.doc+")"));return d}
function detail(n){var tr=el("tr","det"),td=el("td");td.colSpan=cols.length;
 var c=el("div","cl");c.appendChild(el("b",null,"token groups among the top "+n.examples.length+": "));c.appendChild(document.createTextNode(n.clusters.map(function(x){return x.token+" ×"+x.count+" (max "+x.max.toFixed(2)+")"}).join(" · ")));td.appendChild(c);
 n.examples.forEach(function(e){td.appendChild(example(e))});tr.appendChild(td);return tr}
function render(){var q=document.getElementById("q").value.trim(),ly=sel.value;
 var rows=ns.filter(function(n){return (ly===""||n.layer===+ly)&&match(n,q)});
 rows.sort(function(a,b){var x=a[st.key],y=b[st.key];var r=x<y?-1:x>y?1:(a.layer-b.layer||a.neuron-b.neuron);return st.asc?r:-r});
 Array.prototype.forEach.call(head.children,function(th){th.className=th.dataset.k===st.key?(st.asc?"on asc":"on"):""});
 body.textContent="";var lim=Math.min(rows.length,500);
 for(var i=0;i<lim;i++)(function(n){var tr=el("tr","row");tr.appendChild(el("td","num",n.layer));tr.appendChild(el("td","num",n.neuron));
  [n.max,n.mean,n.std,n.active_frac,n.sparsity].forEach(function(v){tr.appendChild(el("td","num",v.toFixed(3)))});tr.appendChild(el("td",null,topTok(n)));
  var d=null;tr.onclick=function(){if(d){d.remove();d=null}else{d=detail(n);tr.after(d)}};body.appendChild(tr)})(rows[i]);
 document.getElementById("count").textContent=rows.length+" of "+ns.length+" neurons"+(rows.length>lim?" (first "+lim+" shown)":"")}
document.getElementById("q").oninput=render;sel.onchange=render;
document.getElementById("reset").onclick=function(){document.getElementById("q").value="";sel.value="";st={key:"max",asc:false};render()};
render();
})();
</script></body></html>
"""


def render_html(atlas: dict) -> str:
    data = json.dumps(atlas, separators=(",", ":"), ensure_ascii=True)
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _HTML.replace("__DATA__", data)


# ----------------------------------------------------------------------------- query (stdlib only)
def load_atlas(path: str | Path) -> dict:
    try:
        a = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AtlasError(f"cannot read atlas {path}: {e}")
    if a.get("schema") != SCHEMA:
        raise AtlasError(f"{path}: not a {SCHEMA} file")
    return a


def query_atlas(atlas: dict, token=None, exact=False, neuron=None, layer=None, sort="max", limit=10) -> list[dict]:
    rows = atlas["neurons"]
    if layer is not None:
        rows = [n for n in rows if n["layer"] == layer]
    if neuron:
        (l, i), = parse_neurons(neuron)[:1]
        rows = [n for n in rows if n["layer"] == l and n["neuron"] == i]
    if token:
        t = token.lower().strip() if not exact else token
        def hit(n):
            toks = [c["token"] for c in n["clusters"]] + [e["token"].strip().lower() for e in n["examples"]]
            return any((x == t) if exact else (t in x) for x in toks)
        rows = [n for n in rows if hit(n)]
    rev = sort != "sparsity_asc"
    key = "sparsity" if sort == "sparsity_asc" else sort
    return sorted(rows, key=lambda n: (n[key], -n["layer"], -n["neuron"]), reverse=rev)[:limit]


def format_row(n: dict, examples: int = 0) -> str:
    toks = ", ".join(f"{c['token']!r}x{c['count']}" for c in n["clusters"][:5])
    s = f"L{n['layer']}.N{n['neuron']}  max={n['max']:.3f} mean={n['mean']:.3f} sparsity={n['sparsity']:.3f}  top tokens: {toks}"
    for e in n["examples"][:examples]:
        s += f"\n    {e['act']:7.3f}  {(e['before'] + '[' + e['token'] + ']' + e['after'])!r}"
    return s


# ----------------------------------------------------------------------------- fetch
def fetch_urls(model: str) -> list[tuple[str, str]]:
    base = f"https://huggingface.co/{model}/resolve/main/"
    return [(base + f, f) for f in ("config.json", "vocab.json", "merges.txt", "model.safetensors")]


def cmd_fetch(a) -> int:
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for url, name in fetch_urls(a.model):
        target = dest / name
        if target.exists() and target.stat().st_size > 0 and not a.force:
            print(f"have {target}")
            continue
        print(f"get  {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": f"neuron-atlas/{__version__}"}), timeout=60) as r, open(tmp, "wb") as f:
                while True:
                    blk = r.read(1 << 20)
                    if not blk:
                        break
                    f.write(blk)
        except Exception as e:
            raise AtlasError(f"download of {url} failed: {e}")
        tmp.replace(target)
    print(f"done: {dest} (model {a.model}; check its license on the model page before reuse)")
    return 0


# ----------------------------------------------------------------------------- CLI
def cmd_build(a) -> int:
    docs = load_corpus(a.corpus)
    atlas = build_atlas(a.model_dir, docs, layers=a.layers, top_k=a.top_k, context=a.context, top_neurons=a.top_neurons,
                        neurons=parse_neurons(a.neurons) if a.neurons else None, seq_len=a.seq_len, active_threshold=a.active_threshold,
                        model_name=a.model_name, corpus_name=Path(a.corpus).name if a.corpus else "bundled")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "atlas.json").write_text(json.dumps(atlas, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "atlas.html").write_text(render_html(atlas), encoding="utf-8")
    m = atlas["meta"]
    print(f"atlas: {m['neurons_in_atlas']} neurons, {m['corpus_tokens']} corpus tokens, {m['run']['seconds_total']} s "
          f"({m['run']['seconds_forward']} s forward) -> {out / 'atlas.html'}, {out / 'atlas.json'}")
    return 0


def cmd_query(a) -> int:
    atlas = load_atlas(a.atlas)
    rows = query_atlas(atlas, token=a.token, exact=a.exact, neuron=a.neuron, layer=a.layer, sort=a.sort, limit=a.limit)
    if not rows:
        print("no neuron matches")
        return 0
    for n in rows:
        print(format_row(n, a.examples))
    return 0


def cmd_corpus(a) -> int:
    print("\n\n".join(bundled_documents()))
    return 0


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="neuron_atlas.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download a GPT-2-class model (config, vocab, merges, safetensors) from huggingface.co")
    f.add_argument("--model", default="distilgpt2"); f.add_argument("--dest", required=True); f.add_argument("--force", action="store_true")
    b = sub.add_parser("build", help="run the corpus through the model and write atlas.html + atlas.json")
    b.add_argument("--model-dir", required=True); b.add_argument("--model-name"); b.add_argument("--out", default="atlas-out")
    b.add_argument("--layers", default="all", help="all | 5 | 4,5 | 0-5"); b.add_argument("--top-k", type=int, default=10)
    b.add_argument("--context", type=int, default=8, help="tokens of context each side"); b.add_argument("--top-neurons", type=int, default=100, help="neurons kept per layer, ranked by max activation (0 = all)")
    b.add_argument("--neurons", help="explicit list, e.g. 5.12,5.77 (overrides --top-neurons)"); b.add_argument("--corpus", help="text file, documents separated by blank lines (default: bundled)")
    b.add_argument("--seq-len", type=int, default=128); b.add_argument("--active-threshold", type=float, default=0.5)
    q = sub.add_parser("query", help="search an atlas.json (stdlib only)")
    q.add_argument("atlas"); q.add_argument("--token"); q.add_argument("--exact", action="store_true"); q.add_argument("--neuron"); q.add_argument("--layer", type=int)
    q.add_argument("--sort", default="max", choices=["max", "mean", "std", "active_frac", "sparsity", "sparsity_asc"]); q.add_argument("--limit", type=int, default=10); q.add_argument("--examples", type=int, default=0)
    sub.add_parser("corpus", help="print the bundled corpus")
    a = ap.parse_args(argv)
    try:
        return {"fetch": cmd_fetch, "build": cmd_build, "query": cmd_query, "corpus": cmd_corpus}[a.cmd](a)
    except AtlasError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
