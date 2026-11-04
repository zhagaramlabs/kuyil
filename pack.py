"""Kuyil data prep, part 2: the aligner's TextGrids + features -> one training pack for train.py.

    python pack.py data/indictts data/indictts.pt

Each utterance becomes tokens [<bos>, phones and breaks ..., <eos>] with a frame count per token. Silence
between two words goes to the break token there (<space> or punctuation). Edge silence is trimmed to 3 frames
and given to <bos> and to <eos> (or to a final punctuation mark), as in PiVo's edge-corrected corpus.
Utterances whose alignment doesn't match our phones are dropped and counted.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from frontend import SYMBOLS
from prepare import mfa_phone

MARGIN = 3
ID = {s: i for i, s in enumerate(SYMBOLS)}
INTERVAL = re.compile(r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "(.*?)"')


def phone_intervals(path):
    for item in path.read_text(encoding="utf-8").split("item [")[1:]:
        name = re.search(r'name = "(.*?)"', item)
        if name and name.group(1).split()[-1] == "phones":
            return [(round(float(a) * 100), round(float(b) * 100), t) for a, b, t in INTERVAL.findall(item)
                    if t not in ("", "sil", "sp", "spn", "<eps>")]
    return []


def align(words, intervals, n_frames):
    """-> tokens, durations and the feature crop (start, end), or a reason string."""
    flat = [p for w in words if isinstance(w, list) for p in w]
    if [mfa_phone(p) for p in flat] != [t for _, _, t in intervals]:
        return "phones differ"
    tokens, durs, k, prev_end = ["<bos>"], [0], 0, None
    for w in words:
        if not isinstance(w, list):
            tokens.append(w); durs.append(0)
            continue
        for p in w:
            start, end, _ = intervals[k]
            if prev_end is not None and start > prev_end:  # silence before this phone
                durs[-1] += start - prev_end  # a break token if one is there, else the previous phone
            tokens.append(p); durs.append(end - start)
            prev_end, k = end, k + 1
    lead = min(MARGIN, intervals[0][0])
    tail = min(MARGIN, max(0, n_frames - intervals[-1][1]))
    durs[0] = lead
    if tokens[-1] in SYMBOLS[4:10]:
        durs[-1] += tail
        tail_eos = 0
    else:
        tail_eos = tail
    tokens.append("<eos>"); durs.append(tail_eos)
    start = intervals[0][0] - lead
    end = start + sum(durs)
    if end > n_frames:
        return "longer than features"
    if any(d < 1 for t, d in zip(tokens, durs) if t not in SYMBOLS[:10]):
        return "phone with no frames"
    return tokens, durs, (start, end)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    utts, why = [], Counter()
    grids = {p.stem: p for p in (src / "aligned").rglob("*.TextGrid")}
    for line in open(src / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        tg = grids.get(r["id"])
        if not tg:
            why["not aligned"] += 1
            continue
        feat = np.load(src / "feat" / f"{r['id']}.npy")
        a = align(r["words"], phone_intervals(tg), len(feat))
        if isinstance(a, str):
            why[a] += 1
            continue
        tokens, durs, (s, e) = a
        utts.append({"id": r["id"], "voice": r["voice"], "split": r["split"],
                     "tokens": torch.tensor([ID[t] for t in tokens], dtype=torch.int16),
                     "durations": torch.tensor(durs, dtype=torch.int16),
                     "features": torch.from_numpy(feat[s:e]).half()})
    train = torch.cat([u["features"].float() for u in utts if u["split"] == "train"])
    torch.save({"symbols": SYMBOLS, "voices": ["female", "male"], "feature_mean": train.mean(0),
                "feature_std": train.std(0), "utterances": utts}, out)
    frames = sum(len(u["features"]) for u in utts)
    print(f"{len(utts)} utterances, {frames} frames ({frames / 360000:.2f} h); dropped: {dict(why)}")
