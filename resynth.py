"""Vocoder ceiling check: real speech -> PiVo's 20 vocoder features -> PiVo's FARGAN vocoder -> audio.

Each input becomes 16 kHz mono 16-bit PCM (ffmpeg). Opus's own extractor (fargan_demo -features, the
path PiVo trained on) turns it into 20 features per 10 ms: 18 cepstral coefficients, pitch
(log2(256/period) - 1.5) and pitch correlation. PiVo's released vocoder graphs and 0.85 de-emphasis
turn them back into audio, the same way PiVo's runtime does. Writes <name>.orig.wav (the 16 kHz input)
and <name>.resynth.wav, sample-aligned, and prints a log-spectral distance, plus band-energy and pitch
agreement from re-analysing the resynthesis with the same extractor.

  python resynth.py in1.wav [in2.wav ...] --out results/fargan_check

Needs ffmpeg on PATH and (local copies in models/ are the defaults):
  FARGAN_DEMO or --fargan-demo: fargan_demo from github.com/xiph/opus at bafd40dcbd50 (PiVo's pinned
    revision), built with ./autogen.sh && ./configure --enable-deep-plc && make
  PIVO_DIR or --pivo: a snapshot of huggingface.co/empero-ai/PiVo-1.4M at 515d84f (uses
    onnx/conditioning.onnx and onnx/vocoder_step.onnx)
Use fargan_demo for analysis only: its C synthesis weights (fargan_data.c) are a different FARGAN from
PiVo's, which is Opus's PyTorch checkpoint fargan_sq1Ab_adv_50.pth.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import lfilter, stft

STATES = {"memory": 256, "gru1": 160, "gru2": 128, "gru3": 128, "frame_state": 164}


def load16k(path):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, "<i2")


def features(pcm, fargan_demo):
    with tempfile.TemporaryDirectory() as d:
        pcm.astype("<i2").tofile(f"{d}/in.pcm")
        subprocess.run([fargan_demo, "-features", f"{d}/in.pcm", f"{d}/f.f32"], check=True)
        f = np.fromfile(f"{d}/f.f32", "<f4").reshape(-1, 20)
    # Feature k describes samples [160k-80, 160k+80); PiVo's training audio uses the same framing.
    assert len(f) == len(pcm) // 160 and np.isfinite(f).all()
    return f


def periods(f):
    return np.rint(256 / 2 ** (np.clip(f[..., 18], -10, 10) + 1.5)).clip(32, 255).astype(np.int64)


def vocode(f, cond_net, step):
    """PiVo's runtime loop: frame k is conditioned on frames k-1, k, k+1 (edges repeated) and gives 4 x 40 samples."""
    ctx = np.concatenate([f[:1], f, f[-1:]])[None].astype(np.float32)
    p = periods(ctx)
    cond = cond_net.run(None, {"features": ctx, "periods": p})[0][0]
    state = {k: np.zeros((1, n), np.float32) for k, n in STATES.items()}
    out = []
    for k in range(len(f)):
        for s in range(4):
            r = step.run(None, {"conditioning": cond[k : k + 1, 80 * s : 80 * (s + 1)], "period": p[:, k + 1], **state})
            state = dict(zip(STATES, r[1:]))
            out.append(r[0][0])
    return lfilter([1], [1, -0.85], np.concatenate(out))


def lsd(a, b):
    """Log-spectral distance in dB, over frames within 50 dB of the loudest frame of a."""
    la, lb = (10 * np.log10(np.abs(stft(x, nperseg=512, noverlap=352)[2]) ** 2 + 1e-10) for x in (a, b))
    e = 10 * np.log10((10 ** (la / 10)).sum(0))
    keep = e > e.max() - 50
    return np.sqrt(((la - lb)[:, keep] ** 2).mean(0)).mean()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    models = Path(__file__).resolve().parent / "models"  # gitignored local copies
    ap.add_argument("--fargan-demo", default=os.environ.get("FARGAN_DEMO", models / "fargan/fargan_demo"))
    ap.add_argument("--pivo", default=os.environ.get("PIVO_DIR", models / "pivo-1.4m"))
    a = ap.parse_args()
    assert a.fargan_demo and a.pivo, "set FARGAN_DEMO and PIVO_DIR, or pass --fargan-demo and --pivo"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    cond_net, step = (ort.InferenceSession(f"{a.pivo}/onnx/{n}.onnx", opts, providers=["CPUExecutionProvider"])
                      for n in ("conditioning", "vocoder_step"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("clip\tseconds\tlsd_db\tcep_db\tvoiced_frames\tpitch_median_cents\tpitch_off_over_1_semitone\t"
          "voicing_added\tvoicing_lost")
    for src in map(Path, a.inputs):
        pcm = load16k(src)
        f = features(pcm, a.fargan_demo)
        assert len(f) >= 2, f"{src} is too short"
        # The vocoder's output lags the source by half a frame (80 samples).
        y = np.round(np.clip(vocode(f, cond_net, step)[80:], -1, 1) * 32767).astype("<i2")
        x = pcm[: len(y)]
        sf.write(out / f"{src.stem}.orig.wav", x, 16000, subtype="PCM_16")
        sf.write(out / f"{src.stem}.resynth.wav", y, 16000, subtype="PCM_16")
        # Re-analyse the resynthesis. cep_db: RMS error of the 18 band log-energies, over frames within 50 dB
        # of the loudest (c0 is sqrt(18) x the mean log10 band energy). Pitch: frames voiced (pitch
        # correlation > 0.1) in both. voicing_added/lost: share of frames voiced only after/before.
        fx, fy = f[: len(y) // 160], features(y, a.fargan_demo)
        loud = fx[:, 0] > fx[:, 0].max() - 5 * np.sqrt(18)
        cep = 10 * np.sqrt(((fx[loud, :18] - fy[loud, :18]) ** 2).mean(1)).mean()
        vx, vy = fx[:, 19] > 0.1, fy[:, 19] > 0.1
        v = vx & vy
        cents = 1200 * np.abs(fx[v, 18] - fy[v, 18])
        print(f"{src.stem}\t{len(x) / 16000:.1f}\t{lsd(x / 32768, y / 32768):.2f}\t{cep:.2f}\t{v.sum()}\t"
              f"{np.median(cents):.0f}\t{(cents > 100).mean():.1%}\t{(vy & ~vx).mean():.1%}\t{(vx & ~vy).mean():.1%}", flush=True)
