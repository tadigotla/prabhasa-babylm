#!/usr/bin/env python3
"""Build the gold kāraka role-contrast cache (H1_CONTRAST, ADR-0043, task 5.4).

Realizes the gold contrast corpus (active + passive per frame) and writes a
trainer-ready cache. Unlike ``build_shabdabodha_cache`` (spaCy parses, label
noise), every role here is gold-by-construction from the frame realizer.

Outputs, into ``<out-dir>``:
  - english_base.txt                     the corpus lines
  - english_base.bin                     uint16 token ids (per-line EncodeAsIds, no eos)
  - shabdabodha_roles_eos.bin            uint8 per-token gold role ids (+ eos role / line)
  - shabdabodha_roles_eos_shuffled.bin   the ADR-0043 shuffled-role specificity control

With ``--babylm <txt>`` the contrast lines are mixed into the BabyLM lines
(Option B): gold roles on contrast tokens, ``none`` on BabyLM tokens, so MLM
trains on real English while the aux head learns roles only on the contrast
slice. CPU-only.

    uv run python scripts/build_contrast_cache.py \\
        --out-dir data/corpora/contrast_gold --n-frames 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm

from psalm.domain.data.karaka_frames import enumerate_frames
from psalm.infrastructure.ml.contrast_corpus import (
    corpus_with_roles,
    flatten_corpus,
    mix_lines,
    none_role_lines,
    shuffle_roles,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="data/tokenizer/strict_small/spm.model")
    ap.add_argument("--n-frames", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--babylm", default=None, help="optional BabyLM .txt to mix in (Option B)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)

    frames = list(enumerate_frames(args.n_frames, seed=args.seed))
    contrast = corpus_with_roles(frames, sp.EncodeAsPieces)
    n_contrast = len(contrast)

    if args.babylm:
        bg_lines = Path(args.babylm).read_text(encoding="utf-8").splitlines()
        background = none_role_lines(bg_lines, sp.EncodeAsPieces)
        pairs = mix_lines(contrast, background, seed=args.seed)
        print(f"mixed corpus: {n_contrast:,} contrast + {len(background):,} background lines")
    else:
        pairs = contrast
        print(f"contrast-only corpus: {n_contrast:,} lines")

    lines, roles = flatten_corpus(pairs, with_eos_role=True)

    token_ids: list[int] = []
    for line in lines:
        token_ids.extend(sp.EncodeAsIds(line))
    tok_arr = np.array(token_ids, dtype=np.uint16)
    role_arr = np.array(roles, dtype=np.uint8)

    # Hard alignment invariant (mirrors build_shabdabodha_cache): the eos-role
    # stream has one extra 'separator' per line over the no-eos token stream.
    expected = len(tok_arr) + len(lines)
    if len(role_arr) != expected:
        raise SystemExit(
            f"alignment bug: roles={len(role_arr):,} tokens={len(tok_arr):,} "
            f"lines={len(lines):,} expected={expected:,}"
        )

    (out / "english_base.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tok_arr.tofile(out / "english_base.bin")
    role_arr.tofile(out / "shabdabodha_roles_eos.bin")
    shuffled = np.array(shuffle_roles(roles, seed=args.seed), dtype=np.uint8)
    shuffled.tofile(out / "shabdabodha_roles_eos_shuffled.bin")

    print(
        f"wrote lines={len(lines):,} tokens={len(tok_arr):,} roles={len(role_arr):,} "
        f"-> {out} (alignment OK)"
    )


if __name__ == "__main__":
    main()
