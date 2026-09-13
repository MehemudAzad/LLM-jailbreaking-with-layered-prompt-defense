"""Layer 1.5 -- structural cipher check (OUR design; bonus-eligible).

Why this layer exists
---------------------
M3 measured Layer 1 (gpt2-large windowed perplexity, threshold 424.8) against the
frozen battery and found it catches exactly the *character-level* obfuscations --
`rot13`, `leetspeak`, `disemvowel` -- and misses `base64`. That is not a tuning
failure: a base64 blob is high-entropy but its byte-pair tokens are individually
common, so its windowed perplexity sits inside the benign range. Lowering the
threshold far enough to catch it costs benign false positives, which defeats the
purpose of the 0%-FPR calibration.

So the gap is structural, not statistical, and the fix is a structural test:
a base64 payload is not *unusual English*, it is *not English at all*, and that is
cheap to decide exactly rather than probabilistically.

What it does
------------
Scan whitespace-delimited tokens for encoded payloads:

  base64 : >= `min_b64_len` chars from the base64 alphabet, length a multiple of 4,
           decoding to mostly-printable ASCII. The decode test is what keeps the
           false-positive rate at zero -- a long ordinary word is not valid base64,
           and a long valid-base64 string is not an ordinary word.
  hex    : >= `min_hex_len` contiguous hex digits (an even count).

Cost: one regex pass, no model, no GPU, sub-millisecond. It runs after L1 so the
transcript records L1's perplexity score for every prompt either way (the report
needs L1's score even on prompts L1.5 ends up blocking).

Deliberately NOT caught here: rot13/leetspeak/disemvowel (L1 already gets those) and
fluent natural-language attacks (persona, prefix injection) -- those are not encoded,
they are English, and they are L3/L4's problem.

Config (`[defense.layer1_5_structural]`): `enabled`, `enforce`, `min_b64_len`,
`min_hex_len`.
"""
from __future__ import annotations

import base64
import binascii
import re

from defense.base import Action, DefenseContext, DefenseLayer, Stage, Verdict

_B64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
# strip the punctuation a payload usually arrives wrapped in ("...blob.", (blob), 'blob')
_STRIP = "\"'`.,;:!?()[]{}<>"


def _printable_ratio(raw: bytes) -> float:
    if not raw:
        return 0.0
    ok = sum(1 for b in raw if 32 <= b <= 126 or b in (9, 10, 13))
    return ok / len(raw)


def _b64_payload(token: str, min_len: int) -> str | None:
    """Return the decoded text if `token` is a real base64 payload, else None."""
    if len(token) < min_len or len(token) % 4 != 0 or not _B64_RE.match(token):
        return None
    try:
        raw = base64.b64decode(token, validate=True)
    except (binascii.Error, ValueError):
        return None
    # A decode that yields binary noise is a false alarm (some long words decode
    # "successfully" into garbage); a real smuggled instruction is readable text.
    if _printable_ratio(raw) < 0.90:
        return None
    return raw.decode("utf-8", errors="replace")


def _hex_payload(token: str, min_len: int) -> str | None:
    if len(token) < min_len or len(token) % 2 != 0 or not _HEX_RE.match(token):
        return None
    try:
        raw = bytes.fromhex(token)
    except ValueError:
        return None
    if _printable_ratio(raw) < 0.90:
        return None
    return raw.decode("utf-8", errors="replace")


class StructuralCipherCheck(DefenseLayer):
    name = "layer1_5_structural"
    stage = Stage.PRE
    source = "own design (closes the base64 gap measured in M3)"

    def process(self, ctx: DefenseContext) -> DefenseContext:
        enforce = bool(self.config.get("enforce", True))
        min_b64 = int(self.config.get("min_b64_len", 24))
        min_hex = int(self.config.get("min_hex_len", 32))

        hit_kind = hit_token = decoded = None
        for token in ctx.prompt.split():
            token = token.strip(_STRIP)
            decoded = _b64_payload(token, min_b64)
            if decoded is not None:
                hit_kind, hit_token = "base64", token
                break
            decoded = _hex_payload(token, min_hex)
            if decoded is not None:
                hit_kind, hit_token = "hex", token
                break

        if hit_kind is None:
            ctx.record(Verdict(self.name, Action.ALLOW, reason="no encoded payload found"))
            return ctx

        action = Action.BLOCK if enforce else Action.ALLOW
        preview = decoded[:80].replace("\n", " ")
        ctx.record(Verdict(
            self.name, action,
            reason=f"{hit_kind} payload ({len(hit_token)} chars) decodes to readable text; enforce={enforce}",
            detail={"kind": hit_kind, "token_len": len(hit_token), "decoded_preview": preview},
        ))
        return ctx
