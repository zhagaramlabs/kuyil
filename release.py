"""Pack a trained run into a Hugging Face model folder: weights, settings, PiVo's vocoder with its notices,
NOTICE, a model card and two voice samples. It checks that the folder speaks exactly as the checkpoint does.

    python release.py 5m            # -> release/kuyil-5m/, then: hf upload zhagaramlabs/kuyil-5m release/kuyil-5m
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file

from synth import PIVO, load

# The best seed of each size (lowest validation loss) and its test-set scores, from lab notes 2026-11-04:
# IndicConformer-600M on the 125 held-out sentences; the original recordings score WER 19.2%, CER 2.9%.
SIZES = {
    "1.4m": dict(run="1.4m-s44", total="1,401,562", wer="17.6%", cer="3.0%", name="1.4M"),
    "5m": dict(run="5m-s42", total="4,892,394", wer="18.4%", cer="3.0%", name="5M"),
    "20m": dict(run="20m-s43", total="22,651,402", wer="18.3%", cer="2.9%", name="20M"),
}
SAMPLE = "வணக்கம், நான் குயில். உங்களுக்கு என்ன உதவி வேண்டும்?"
IITM = ("COPYRIGHT 2016 TTS Consortium, TDIL, Meity represented by Hema A Murthy & S Umesh, DEPARTMENT OF "
        "Computer Science and Engineering and Electrical Engineering, IIT Madras. ALL RIGHTS RESERVED")


def card(size, s, ck, hours):
    return f"""---
license: cc-by-4.0
language:
- ta
- en
pipeline_tag: text-to-speech
tags:
- tamil
- tanglish
- tts
- tiny
- pivo
- fargan
datasets:
- SPRINGLab/IndicTTS_Tamil
---

# Kuyil {s['name']} (குயில்)

A tiny, fast Tamil text-to-speech model: {s['total']} parameters in all, with the vocoder. It speaks
about 15 times faster than real time on one CPU thread, so it suits phones, screen readers and voice
agents. It reads Tamil script, English words in Latin script, and numbers and ₹ amounts as people say them.

Made by [Zhagaram Labs](https://zhagaramlabs.com), a non-profit Tamil-first AI lab. Code:
[github.com/zhagaramlabs/kuyil](https://github.com/zhagaramlabs/kuyil). Other sizes:
[kuyil-1.4m](https://huggingface.co/zhagaramlabs/kuyil-1.4m) ·
[kuyil-5m](https://huggingface.co/zhagaramlabs/kuyil-5m) ·
[kuyil-20m](https://huggingface.co/zhagaramlabs/kuyil-20m).

## தமிழில்

குயில், தமிழ் உரையைப் பேச்சாக மாற்றும் ஒரு சிறிய, வேகமான மாதிரி. குரல் உருவாக்கியையும் (vocoder) சேர்த்து,
இதில் {s['total']} அளவுருக்கள் (parameters) உள்ளன. ஒரு CPU thread-இலேயே நிகழ்நேரத்தைவிட சுமார் 15 மடங்கு
வேகமாகப் பேசும். அதனால் கைபேசிகள், திரை வாசிப்பான்கள் (screen readers), குரல் உதவியாளர்கள் ஆகியவற்றுக்கு
ஏற்றது. தமிழ் எழுத்து, ஆங்கில எழுத்தில் வரும் ஆங்கிலச் சொற்கள், எண்கள், ₹ தொகைகள் ஆகியவற்றைப் பேசும்.

- **குரல்கள்:** பெண் (`female`), ஆண் (`male`). இவை சென்னை ஐஐடியின் (IIT Madras) IndicTTS தமிழ்த்
  தரவுத்தொகுப்பில் பேசிய இருவரின் குரல்கள்.
- **தெளிவு:** பயிற்சியில் சேர்க்காத 125 வாக்கியங்களில், பேச்சு அறிதல் மாதிரியின் (IndicConformer)
  எழுத்துப் பிழை விகிதம், குயிலின் பேச்சில் {s['cer']}, மூல ஒலிப்பதிவுகளில் 2.9%. அதாவது மூலப் பதிவுகள்
  அளவுக்கே தெளிவு.
- **வரம்புகள்:** இலக்கிய நடை வாசிப்பில் மட்டுமே பயிற்சி பெற்றது. உரையாடல் தொனியும், தங்கிலீஷ்
  பேச்சுப் பயிற்சியும் இன்னும் இல்லை.
- **உரிமம்:** CC BY 4.0. பயன்படுத்தும்போது Zhagaram Labs-ஐயும், சென்னை ஐஐடியின் IndicTTS-ஐயும்
  குறிப்பிடுங்கள். இந்தக் குரல்கள் உண்மையான மனிதர்களுடையவை. அவர்களைப் போல் நடித்து ஏமாற்றப்
  பயன்படுத்தாதீர்கள்.

## Listen

"{SAMPLE}"

| Voice | Sample |
|---|---|
| female | <audio controls src="https://huggingface.co/zhagaramlabs/kuyil-{size}/resolve/main/samples/female.wav"></audio> |
| male | <audio controls src="https://huggingface.co/zhagaramlabs/kuyil-{size}/resolve/main/samples/male.wav"></audio> |

## Use it

```bash
git clone https://github.com/zhagaramlabs/kuyil && cd kuyil && uv sync
hf download zhagaramlabs/kuyil-{size} --local-dir kuyil-{size}
uv run python synth.py kuyil-{size} "வணக்கம், நான் குயில்." --voice female --out hello.wav
```

In Python, from the `kuyil` folder: `speak, voices = synth.load("kuyil-{size}")`, then
`speak(text, "female")` returns 16 kHz audio as a NumPy array.

## How well it works

The 125 held-out IndicTTS sentences, spoken by each size and transcribed by IndicConformer-600M, compared
with the original recordings. These are literary sentences full of long compound words, so word error is
high even for the recordings, and character error is the steadier measure.

| Audio | Character error | Word error |
|---|---|---|
| Original recordings | 2.9% | 19.2% |
| **This model** (run `{s['run']}`) | **{s['cer']}** | **{s['wer']}** |

All three sizes are as intelligible as the recordings, so a recognizer can't choose between them.
Native listeners will. Synthesis runs at a real-time factor of 0.06–0.07 on one Apple-silicon CPU
thread, vocoder included, at every size.

## How it was made

- **Recipe:** [PiVo-1.4M](https://huggingface.co/empero-ai/PiVo-1.4M)'s, rebuilt for Tamil. A text encoder
  and duration predictor, then a causal decoder that predicts 20 FARGAN features every 10 ms. PiVo's own
  FARGAN vocoder (in `pivo/`) turns those into audio.
- **Settings:** {json.dumps(ck['config'])}.
- **Data:** IndicTTS Tamil from IIT Madras, via the
  [SPRINGLab/IndicTTS_Tamil](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil) mirror. That's
  9,437 sentences from one female and one male speaker. After edge trimming, 24.8 h are left. Training
  used 9,022 utterances of up to 30 s.
- **Alignment:** phone timings from Montreal Forced Aligner 3.4.3, trained on our own Tamil phone set.
- **Training:** smooth-L1 on normalized features plus 0.1 × squared error on log durations. Batch 8,
  learning rate 3e-4, 90 epochs on one H100 (about {hours:.1f} h, sharing the GPU with 8 other runs).
  These weights come from epoch {ck['epoch']}, which had the lowest validation loss ({ck['val']:.4f}).
  PiVo's adversarial refinement wasn't done.
- **Text frontend** (`frontend.py` in the code repo): Tamil letters to 74 phones, with the stop
  allophony of spoken Tamil and doubled consonants. English in Latin script goes through CMUdict mapped
  to Tamil phones. Numbers are read the way people say them.

## Limits

- Two voices only, both from studio reading of literary Tamil. Conversational speech will sound read.
- Tanglish is handled by the text frontend, but the model has never heard Tanglish speech.
- Naturalness hasn't been rated by native listeners yet.

## Use it responsibly

These are the voices of two real people, recorded for IIT Madras's TTS database. Don't use them to
impersonate anyone, and say that the speech is synthetic where it matters.

## Licence and attribution

- **These weights:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Credit Zhagaram Labs, and
  IndicTTS from IIT Madras for the training data. Keep the notices in `NOTICE`.
- **Training data:** IndicTTS. The mirror we trained from is labelled CC BY 4.0. Its notice: {IITM}.
- **Vocoder files** in `pivo/`: PiVo-1.4M's FARGAN vocoder, Apache 2.0, with FARGAN-derived material
  under the Xiph/Opus BSD licence. Their licence and notices are kept in `pivo/`.
- **Code:** Apache 2.0, at [github.com/zhagaramlabs/kuyil](https://github.com/zhagaramlabs/kuyil).

```bibtex
@misc{{kuyil2026,
  title  = {{Kuyil: a tiny Tamil text-to-speech model}},
  author = {{{{Zhagaram Labs}}}},
  year   = {{2026}},
  url    = {{https://huggingface.co/zhagaramlabs/kuyil-{size}}}
}}
```
"""


def release(size):
    s = SIZES[size]
    run, out = Path("data/runs") / s["run"], Path("release") / f"kuyil-{size}"
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    hours = json.loads((run / "log.jsonl").read_text().splitlines()[-1])["seconds"] / 3600
    shutil.rmtree(out, ignore_errors=True)
    (out / "pivo/onnx").mkdir(parents=True); (out / "pivo/LICENSES").mkdir(); (out / "samples").mkdir()
    save_file({k: v.contiguous() for k, v in ck["state_dict"].items()}, out / "model.safetensors")
    (out / "config.json").write_text(json.dumps({
        "config": ck["config"], "symbols": ck["symbols"], "voices": ck["voices"],
        "training": {"run": s["run"], "epoch": ck["epoch"], "step": ck["step"], "val": ck["val"],
                     "acoustic_parameters": ck["params"]}}, ensure_ascii=False, indent=1))
    for n in ("conditioning", "vocoder_step"):
        shutil.copy(PIVO / "onnx" / f"{n}.onnx", out / "pivo/onnx")
    for f in ("LICENSE", "NOTICE", "REVISION", "LICENSES/Opus-BSD-3-Clause.txt"):
        shutil.copy(PIVO / f, out / "pivo" / f)
    (out / "NOTICE").write_text(
        f"Kuyil {s['name']}, Zhagaram Labs, 2026. Weights: CC BY 4.0.\n\n"
        f"Trained on IndicTTS Tamil (IIT Madras), via the SPRINGLab/IndicTTS_Tamil mirror (CC BY 4.0).\n{IITM}\n\n"
        "pivo/ holds PiVo-1.4M's FARGAN vocoder (Apache 2.0; FARGAN-derived material under the Xiph/Opus BSD\n"
        "licence), unchanged, with its LICENSE, NOTICE and LICENSES/ from empero-ai/PiVo-1.4M at pivo/REVISION.\n")
    (out / "README.md").write_text(card(size, s, ck, hours))

    # The folder must speak exactly as the training checkpoint does.
    a, _ = load(run / "best.pt")
    b, voices = load(out)
    for v in voices:
        ya, yb = a(SAMPLE, v), b(SAMPLE, v)
        assert np.array_equal(ya, yb), f"{size} {v}: the released folder doesn't match the checkpoint"
        sf.write(out / "samples" / f"{v}.wav", yb, 16000, subtype="PCM_16")
    print(out, f"{sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB")


if __name__ == "__main__":
    for size in sys.argv[1:] or SIZES:
        release(size)
