"""Lock the published SHA-256 anchors for the released LLM prompts.

`docs/README.md` publishes, for each elicitation arm, the SHA-256 of the prompt
*as loaded* -- read as UTF-8 text under universal newlines, right-stripped, then
with the three run-config tokens substituted at their production values. That
digest is the reproducibility contract for the LLM benchmark: it is the exact
string the models were sent, and a replicator who reproduces it knows their copy
of the prompt is the one the paper used.

Nothing else guards it. A stripped trailing space or a well-meant typo fix in a
prompt file would silently invalidate the documented hash *and* silently change
what a replicator elicits. These tests make that a CI failure instead.

Hashing after text decoding is what makes the anchors invariant to LF/CRLF
checkout, so this passes on Windows and Linux alike.

Update policy: if a prompt legitimately changes, re-derive both the digest and
the character count and update EXPECTED and `docs/README.md` in the same commit.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

DOCS_PROMPTS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "prompts"

# Production run values for the three tokens the harness substitutes at load.
TOKENS = (("lookback_qs", "12"), ("lookback_first", "11"), ("max_horizon", "20"))

EXPECTED = {
    "v8": ("6d2914b8fa9e0a56f0b197820dd93581a44e7dd1181d6b4cc0bdb897afdd03ec", 9642),
    "v8s": ("46f4d9d2b5bcfa27c6e94808577b1e08ece6287e86988e877cc8500fac06f69b", 11673),
}


def _as_loaded(arm: str) -> str:
    path = DOCS_PROMPTS / f"llm_benchmark_{arm}_system.txt"
    text = path.read_text(encoding="utf-8").rstrip()
    for token, value in TOKENS:
        text = text.replace("{" + token + "}", value)
    return text


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_prompt_hash_as_loaded(arm: str) -> None:
    digest, n_chars = EXPECTED[arm]
    loaded = _as_loaded(arm)
    assert len(loaded) == n_chars
    assert hashlib.sha256(loaded.encode("utf-8")).hexdigest() == digest


@pytest.mark.parametrize("arm", sorted(EXPECTED))
def test_every_token_is_substituted(arm: str) -> None:
    """No placeholder may survive loading.

    A renamed or newly-added token would otherwise reach the model as a literal
    `{...}`, and the hash test alone would not say why it broke.
    """
    loaded = _as_loaded(arm)
    for token, _ in TOKENS:
        assert "{" + token + "}" not in loaded
    assert "{" not in loaded and "}" not in loaded
