# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "torch", "transformers"]
# ///
"""One-off check that neuron_atlas.py's numpy forward pass matches HuggingFace transformers (PyTorch, CPU) on the same weights.
Not needed to use neuron-atlas. Usage: uv run samples/validate_against_transformers.py MODEL_DIR [out.json]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch, transformers
import neuron_atlas as na

d = sys.argv[1]
model = na.GPT2.load(d); bpe = na.BPE.from_dir(d)
hf = transformers.GPT2LMHeadModel.from_pretrained(d, torch_dtype=torch.float32).eval()
tok = transformers.GPT2TokenizerFast.from_pretrained(d)
docs = na.bundled_documents()
worst_logit = worst_act = 0.0; n_tok_mismatch = n_docs = 0; total = 0
for doc in docs:
    ours = bpe.encode(doc); theirs = tok(doc)["input_ids"]
    if ours != theirs:
        n_tok_mismatch += 1
    ids = [model.bos] + ours[:127]
    with torch.no_grad():
        out = hf(torch.tensor([ids]), output_hidden_states=False)
        hooks, store = [], {}
        for i, blk in enumerate(hf.transformer.h):
            hooks.append(blk.mlp.act.register_forward_hook(lambda m, a, o, i=i: store.__setitem__(i, o[0].numpy())))
        hf(torch.tensor([ids]))
        for h in hooks: h.remove()
    logits = model.logits(ids); _, acts = model.run(ids, layers=range(model.n_layer))
    worst_logit = max(worst_logit, float(np.abs(logits - out.logits[0].numpy()).max()))
    worst_act = max(worst_act, max(float(np.abs(acts[i] - store[i]).max()) for i in range(model.n_layer)))
    n_docs += 1; total += len(ids)
res = {"docs": n_docs, "tokens_compared": total, "tokenizer_mismatching_docs": n_tok_mismatch, "max_abs_logit_diff": worst_logit,
       "max_abs_mlp_activation_diff": worst_act, "torch": torch.__version__, "transformers": transformers.__version__}
print(json.dumps(res, indent=1))
if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
