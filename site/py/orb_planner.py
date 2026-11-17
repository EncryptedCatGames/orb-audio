"""The ORB planner in plain numpy, for the browser.

A line-for-line port of ``orb.model.planner.Planner`` (pre-LN GPT blocks,
exact-erf GELU, tied head) plus ``orb.api.plan``'s decoding loop: greedy,
masked by ``orb.model.spec_tokenizer.SpecGrammar`` at every step, with a KV
cache so each new token costs one row of compute. Weights come from
``site/assets/planner/`` (written by ``scripts/build_site.py`` from
``weights/planner.pt``): one weight blob (int8 rows + fp16 scales since
v0.4, fp16 before) plus a JSON index. Compute is float32.

``scripts/check_web_planner.py`` checks that this decodes the same Spec as
``orb.api.plan`` in torch.
"""

from __future__ import annotations

import json
import time

import numpy as np

LN_EPS = 1e-5


def _erf(x: np.ndarray) -> np.ndarray:
    """erf, float32-accurate (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).

    numpy has no vectorised erf and scipy is a large download in Pyodide.
    """
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x * np.float32(0.7071067811865476)))


def _ln(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


class NumpyPlanner:
    def __init__(self, index: dict, blob: bytes | np.ndarray) -> None:
        cfg = index["config"]
        self.d = int(cfg["d"])
        self.layers = int(cfg["layers"])
        self.heads = int(cfg["heads"])
        self.ctx = int(cfg["ctx"])
        self.vocab = int(cfg["vocab_size"])
        self.w: dict[str, np.ndarray] = {}
        if index.get("dtype") == "int8-rows":
            # Matrices: int8 rows then one fp16 scale per row; vectors: fp16.
            # Byte offsets.
            buf = memoryview(blob).cast("B") if not isinstance(blob, np.ndarray) else blob.view(np.uint8)
            for name, shape, offset, kind in index["tensors"]:
                n = int(np.prod(shape))
                if kind == "q8":
                    q = np.frombuffer(buf, dtype=np.int8, count=n, offset=offset).reshape(shape)
                    sc = np.frombuffer(buf, dtype="<f2", count=shape[0], offset=offset + n)
                    self.w[name] = q.astype(np.float32) * sc.astype(np.float32)[:, None]
                else:
                    self.w[name] = np.frombuffer(buf, dtype="<f2", count=n, offset=offset).astype(np.float32).reshape(shape)
        else:
            raw = np.frombuffer(blob, dtype=np.dtype(index.get("dtype", "float16")))
            for name, shape, offset in index["tensors"]:
                n = int(np.prod(shape))
                self.w[name] = raw[offset : offset + n].astype(np.float32).reshape(shape)
        # Pre-transpose the linear layers once: x @ W.T -> x @ Wt.
        for i in range(self.layers):
            p = f"blocks.{i}."
            for k in ("attn.qkv.weight", "attn.proj.weight", "mlp.0.weight", "mlp.2.weight"):
                self.w[p + k] = np.ascontiguousarray(self.w[p + k].T)
        self.head_t = np.ascontiguousarray(self.w["tok_emb.weight"].T)

    def _block(self, i: int, x: np.ndarray, cache: list) -> np.ndarray:
        """x: (t, d) new positions; cache[i] holds (k, v) of all earlier ones."""
        w, p = self.w, f"blocks.{i}."
        t, d = x.shape
        h, hd = self.heads, d // self.heads
        qkv = _ln(x, w[p + "ln1.weight"], w[p + "ln1.bias"]) @ w[p + "attn.qkv.weight"]
        q = qkv[:, :d].reshape(t, h, hd).transpose(1, 0, 2)
        k = qkv[:, d : 2 * d].reshape(t, h, hd).transpose(1, 0, 2)
        v = qkv[:, 2 * d :].reshape(t, h, hd).transpose(1, 0, 2)
        if cache[i] is not None:
            k = np.concatenate([cache[i][0], k], axis=1)
            v = np.concatenate([cache[i][1], v], axis=1)
        cache[i] = (k, v)
        n = k.shape[1]
        att = (q @ k.transpose(0, 2, 1)) * np.float32(1.0 / np.sqrt(hd))  # (h, t, n)
        if t > 1:  # causal: new row j sees keys up to n - t + j
            mask = np.arange(n)[None, :] > (np.arange(t)[:, None] + n - t)
            att = np.where(mask[None], np.float32(-np.inf), att)
        att = att - att.max(axis=-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(axis=-1, keepdims=True)
        y = (att @ v).transpose(1, 0, 2).reshape(t, d)
        x = x + y @ w[p + "attn.proj.weight"]
        m = _ln(x, w[p + "ln2.weight"], w[p + "ln2.bias"]) @ w[p + "mlp.0.weight"] + w[p + "mlp.0.bias"]
        x = x + _gelu(m) @ w[p + "mlp.2.weight"] + w[p + "mlp.2.bias"]
        return x

    def _step(self, ids: list[int], start: int, cache: list, logits: bool) -> np.ndarray | None:
        x = self.w["tok_emb.weight"][ids] + self.w["pos_emb.weight"][start : start + len(ids)]
        x = x.astype(np.float32)
        for i in range(self.layers):
            x = self._block(i, x, cache)
        if not logits:
            return None
        last = _ln(x[-1:], self.w["ln_f.weight"], self.w["ln_f.bias"])
        return (last @ self.head_t)[0]

    def plan_ids(self, caption: str, max_new_tokens: int = 1024) -> list[int]:
        """Caption -> Spec token ids (greedy, grammar-constrained)."""
        from orb.model.spec_tokenizer import BOS, EOS, NOBED, SEP, SpecGrammar

        prompt = [BOS] + list(caption.encode("utf-8")) + [SEP]
        if len(prompt) >= self.ctx:
            raise ValueError(f"caption too long ({len(prompt)} tokens, context {self.ctx})")
        grammar = SpecGrammar(no_bed=self.vocab > NOBED)
        cache: list = [None] * self.layers
        out: list[int] = []
        allowed = grammar.allowed()
        logits = self._step(prompt, 0, cache, logits=len(allowed) > 1)
        pos = len(prompt)
        for _ in range(max_new_tokens):
            if len(allowed) == 1:
                tok = next(iter(allowed))  # forced by the grammar: argmax of a one-token mask
            else:
                cand = np.fromiter(sorted(allowed), dtype=np.int64)
                tok = int(cand[int(np.argmax(logits[cand]))])
            out.append(tok)
            if tok == EOS:
                break
            grammar.advance(tok)
            allowed = grammar.allowed()
            if pos >= self.ctx:
                raise ValueError("ran out of context")
            logits = self._step([tok], pos, cache, logits=len(allowed) > 1)
            pos += 1
        return out


_PLANNER: NumpyPlanner | None = None


def load(index_json: str, blob) -> dict:
    """Build the planner from the JSON index and the weight blob (bytes or a JS buffer)."""
    global _PLANNER
    t0 = time.perf_counter()
    if hasattr(blob, "to_py"):
        blob = blob.to_py()
    _PLANNER = NumpyPlanner(json.loads(index_json), bytes(blob) if isinstance(blob, memoryview) else blob)
    return {"ms": round((time.perf_counter() - t0) * 1000.0, 1)}


def generate(caption: str) -> dict:
    """Mirror of ``orb.api.generate``'s routing: a ``no bed`` grammar caption is
    read exactly (the planner never saw one); anything else goes to the planner.
    Returns the Spec dict, which path ran and the decode time."""
    from orb.api import spec_to_dict
    from orb.grammar import NO_BED, parse_caption
    from orb.model.spec_tokenizer import SpecTokenizer

    caption = caption.strip()
    try:
        direct = parse_caption(caption)
    except ValueError:  # grammar-shaped but not a valid Spec: the planner reads it
        direct = None
    if direct is not None and direct.texture == NO_BED:
        return {"spec": spec_to_dict(direct), "planned": False, "ms": 0.0, "tokens": 0}
    if _PLANNER is None:
        raise RuntimeError("planner weights are not loaded")
    t0 = time.perf_counter()
    ids = _PLANNER.plan_ids(caption)
    ms = (time.perf_counter() - t0) * 1000.0
    spec = SpecTokenizer().decode_spec(ids)
    return {"spec": spec_to_dict(spec), "planned": True, "ms": round(ms, 1), "tokens": len(ids)}


def needs_planner(caption: str) -> bool:
    """False when ``generate`` can answer without the weights (a ``no bed``
    caption, read exactly)."""
    from orb.grammar import NO_BED, parse_caption

    try:
        direct = parse_caption(caption.strip())
    except ValueError:
        return True
    return direct is None or direct.texture != NO_BED


def generate_js(caption: str) -> str:
    return json.dumps(generate(caption))
