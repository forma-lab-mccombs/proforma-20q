# Documentation

## `release_documentation.pdf`

Detailed technical documentation for the ProForma-20Q benchmark and the Forma
model. Three parts:

| Part | Contents |
|---|---|
| **A. Data** | Sample formation and splits (with the full waterfall), the regularization that maps reported accounting values into model space, the 78-item target universe, its reporting availability, and the accounting identities linking the items. |
| **B. Models** | Forma's training and implementation detail — embeddings, output heads, loss, masking, curriculum, batching, inference, and the seed mixture — plus the exact specification of every competitor (elastic net, random forest, feed-forward networks, chained GBM, pooled fade/AR(1), naive, Chronos-2, LLM panel). |
| **C. LLM benchmark** | The elicitation protocol (inputs, origin sampler, targets, the two prompt arms, models and settings, the decoding-noise subset, the common sample) and the prompts reproduced in full. |

Every number in the document is produced by the pipeline released here. The
underlying Compustat panel is licensed and cannot be redistributed; the release
rebuilds it from a WRDS connection.

## `prompts/`

The byte-exact system prompts used for the LLM benchmark, as loaded by the
harness:

| File | Arm |
|---|---|
| `llm_benchmark_v8_system.txt` | Unstructured (v8) — the arm reported in the paper's headline table for all three models |
| `llm_benchmark_v8s_system.txt` | Structured (v8s) |

Part C of the documentation reproduces these prompts with Unicode box-drawing
and math glyphs transliterated to ASCII for typesetting. **These files are the
authoritative originals**; use them, not the typeset copies, to reproduce the
elicitation exactly.

The harness loads a prompt by reading it as UTF-8 text (universal newlines),
right-stripping it, and substituting three tokens from the run config:
`{lookback_qs}` → `12`, `{lookback_first}` → `11`, `{max_horizon}` → `20` for
the production run. The SHA-256 of the resulting string — the exact text sent to
the models — is:

| Arm | SHA-256 (as loaded, 12/11/20) | chars |
|---|---|---|
| v8 | `6d2914b8fa9e0a56f0b197820dd93581a44e7dd1181d6b4cc0bdb897afdd03ec` | 9,642 |
| v8s | `46f4d9d2b5bcfa27c6e94808577b1e08ece6287e86988e877cc8500fac06f69b` | 11,673 |

Reproduce with:

```python
import hashlib
with open("docs/prompts/llm_benchmark_v8_system.txt", encoding="utf-8", newline=None) as f:
    s = f.read().rstrip()
s = s.replace("{lookback_qs}", "12").replace("{lookback_first}", "11").replace("{max_horizon}", "20")
print(hashlib.sha256(s.encode("utf-8")).hexdigest())
```

Because the hash is taken after universal-newline decoding, it is invariant to
whether the file is checked out with LF or CRLF endings.
`tests/test_prompt_hashes.py` enforces both anchors in CI, so a prompt cannot
drift from its documented hash silently.

> **The elicitation harness itself is not in this repository.** It lives with the
> model code, so the load semantics described above cannot be checked here and
> could in principle drift from the real harness. They are recorded rather than
> executed. (The described substitution is robust to one detail: the only braces
> in either prompt file are the three placeholders, so `str.replace` and
> `str.format` produce identical output.) This mirrors the PDF, which is a build
> artifact of the paper sources and is likewise not regenerable from this
> repository alone.
