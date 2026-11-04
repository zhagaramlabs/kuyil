"""Speak text with a trained Kuyil checkpoint through PiVo's FARGAN vocoder.

    python synth.py CKPT "வணக்கம், நான் குயில்." --voice female --out hello.wav
    python synth.py CKPT --test data/indictts --out data/eval/<run>

The second form speaks every held-out test sentence in its own voice and writes run_asr.py manifests
for the synthesized audio (<out>/kuyil/test.tsv) and the original recordings (<out>/orig/test.tsv):
    cd ../oosai && .venv/bin/python eval/run_asr.py indicconformer --manifest ../kuyil/<out>/kuyil/test.tsv   (and orig)
oosai's eval/tts_samples.py also takes Kuyil (KUYIL_CKPT=... eval/tts_samples.py kuyil) for Tanglish and numbers.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from frontend import phonemize
from model import Config, Kuyil
from prepare import corpus_dir
from resynth import vocode

PIVO = Path(__file__).resolve().parent / "models/pivo-1.4m"


def load(checkpoint, pivo=PIVO):
    """-> speak(text, voice) returning 16 kHz float audio, and the voice names."""
    torch.set_num_threads(1)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Kuyil(Config(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
    cond_net, step = (ort.InferenceSession(f"{pivo}/onnx/{n}.onnx", opts, providers=["CPUExecutionProvider"])
                      for n in ("conditioning", "vocoder_step"))

    def speak(text, voice):
        ids = torch.tensor([ck["symbols"].index(p) for p in ["<bos>"] + phonemize(text) + ["<eos>"]])
        return np.clip(vocode(model.synthesize(ids, ck["voices"].index(voice)).numpy(), cond_net, step), -1, 1)

    return speak, ck["voices"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--voice", default="female")
    ap.add_argument("--test", help="a prepare.py output dir: speak its test split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    speak, voices = load(a.checkpoint)
    if not a.test:
        print(" ".join(phonemize(a.text)))
        t0 = time.time()
        y = speak(a.text, a.voice)
        sf.write(a.out, y, 16000, subtype="PCM_16")
        print(f"{len(y) / 16000:.2f} s of audio in {time.time() - t0:.2f} s: real-time factor {(time.time() - t0) / (len(y) / 16000):.3f}")
    else:
        out, data = Path(a.out), Path(a.test)
        (out / "kuyil").mkdir(parents=True, exist_ok=True); (out / "orig").mkdir(exist_ok=True)
        mk, mo = open(out / "kuyil/test.tsv", "w"), open(out / "orig/test.tsv", "w")
        secs = audio = 0.0
        for line in open(data / "manifest.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r["split"] != "test":
                continue
            voice = voices[r["voice"]]
            t0 = time.time()
            y = speak(r["text"], voice)
            secs += time.time() - t0; audio += len(y) / 16000
            wav = (out / "kuyil" / f"{r['id']}.wav").resolve()
            sf.write(wav, y, 16000, subtype="PCM_16")
            mk.write(f"{r['id']}\t{wav}\t{r['text']}\n")
            mo.write(f"{r['id']}\t{(data / 'corpus' / corpus_dir(voice, r['id']) / (r['id'] + '.wav')).resolve()}\t{r['text']}\n")
        print(f"{audio:.0f} s of test audio in {secs:.0f} s: real-time factor {secs / audio:.3f} (one CPU thread)")
