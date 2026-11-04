"""Kuyil data prep, part 1: IndicTTS Tamil parquet -> 16 kHz WAVs, an aligner corpus, FARGAN features and phones.

    python prepare.py data/indictts_ta data/indictts

Writes under OUT:
  corpus/<voice>-<k>/<id>.wav (16 kHz), <id>.lab  the aligner's corpus; each word is a pronunciation label w<n>
  dictionary.txt                                 w<n> -> aligner phone names (p<symbol index>)
  feat/<id>.npy                                  frames x 20 FARGAN features (fargan_demo -features, as PiVo)
  manifest.jsonl                                 id, voice, split, text, words (phone lists and break tokens)
Then align with the aligner (README, step 2) and run pack.py.
"""
import hashlib, io, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

from frontend import SYMBOLS, words

FARGAN_DEMO = Path(__file__).resolve().parent / "models/fargan/fargan_demo"
VOICES = {0: "female", 1: "male"}  # IndicTTS gender column: one speaker each


def mfa_phone(p):
    return f"p{SYMBOLS.index(p)}"  # opaque names: the aligner lowercases and our set is case-sensitive (t/T, n/N)


def corpus_dir(voice, uid):
    """<voice>-<0..7>: the aligner runs one job per speaker folder, so each voice is split into 8."""
    return f"{voice}-{int(hashlib.sha1(uid.encode()).hexdigest(), 16) % 8}"


def split_of(text):
    """Same text, same split, whichever voice reads it: about 1.5% validation and 1.5% test."""
    h = int(hashlib.sha1(text.encode()).hexdigest(), 16) % 1000
    return "test" if h < 15 else "val" if h < 30 else "train"


def features(wav_path, feat_path):
    pcm = (sf.read(wav_path, dtype="int16")[0]).astype("<i2")
    raw = feat_path.with_suffix(".pcm")
    pcm.tofile(raw)
    f32 = feat_path.with_suffix(".f32")
    subprocess.run([str(FARGAN_DEMO), "-features", str(raw), str(f32)], check=True, capture_output=True)
    f = np.fromfile(f32, "<f4").reshape(-1, 20)
    raw.unlink(); f32.unlink()
    assert len(f) == len(pcm) // 160 and np.isfinite(f).all(), wav_path
    np.save(feat_path, f.astype(np.float32))


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "feat").mkdir(parents=True, exist_ok=True)
    labels, rows, jobs = {}, [], []
    for f in sorted((src / "data").glob("*.parquet")):
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            uid = f"ta{len(rows):05d}"  # the mirror reuses file names: 2,459 of them name two different recordings
            audio, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
            assert sr == 48000 and audio.ndim == 1, (uid, sr)
            voice = VOICES[r["gender"]]
            wav = out / "corpus" / corpus_dir(voice, uid) / f"{uid}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, np.clip(resample_poly(audio, 1, 3), -1, 1), 16000, subtype="PCM_16")
            ws = words(r["text"])
            lab = []
            for w in ws:
                if isinstance(w, list):
                    key = " ".join(w)
                    labels.setdefault(key, f"w{len(labels)}")
                    lab.append(labels[key])
            wav.with_suffix(".lab").write_text(" ".join(lab) + "\n", encoding="utf-8")
            rows.append({"id": uid, "source": f"{f.name}#{i} {r['audio']['path']}", "voice": r["gender"],
                         "split": split_of(r["text"]), "text": r["text"], "words": ws})
            jobs.append((wav, out / "feat" / f"{uid}.npy"))
    with open(out / "dictionary.txt", "w", encoding="utf-8") as d:
        for key, label in labels.items():
            d.write(label + "\t" + " ".join(mfa_phone(p) for p in key.split()) + "\n")
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as m:
        for r in rows:
            m.write(json.dumps(r, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: features(*j), jobs))
    print(len(rows), "utterances,", len(labels), "pronunciations,",
          {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
