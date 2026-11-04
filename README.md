# Kuyil (குயில்): a tiny Tamil and Tanglish voice

Kuyil is a text-to-speech model of a few million parameters for Tamil, including Tamil mixed with
English. It's small and fast enough for phones, screen readers and a voice agent that shouldn't make
people wait. It follows [PiVo](https://huggingface.co/empero-ai/PiVo-1.4M)'s recipe: a small acoustic
model predicts 20 vocoder features per 10 ms, and PiVo's FARGAN vocoder turns them into audio.

Part of [Zhagaram Labs](https://zhagaramlabs.com), a non-profit, Tamil-first open AI lab. Related
repos: [uraikal](https://github.com/zhagaramlabs/uraikal) (the Tamil benchmark and text library Kuyil
reads numbers with) and [oosai](https://github.com/zhagaramlabs/oosai) (the voice agent Kuyil is built for).

**Status (2026-11-04):** steps 0–2 are built. The first size ladder (3 sizes × 3 seeds) is trained on
IndicTTS Tamil and scored: every size is as intelligible as the original recordings ([results](#step-3-listening-and-scoring)).
All three sizes are released on Hugging Face under CC BY 4.0:
[kuyil-1.4m](https://huggingface.co/zhagaramlabs/kuyil-1.4m) ·
[kuyil-5m](https://huggingface.co/zhagaramlabs/kuyil-5m) ·
[kuyil-20m](https://huggingface.co/zhagaramlabs/kuyil-20m). Native listeners pick the default size next.

```bash
hf download zhagaramlabs/kuyil-1.4m --local-dir kuyil-1.4m
uv run python synth.py kuyil-1.4m "வணக்கம், நான் குயில்." --voice female --out hello.wav
```

## Set up

```bash
uv sync                          # Python 3.10+, PyTorch, and uraikal from PyPI
uv run python -m pytest -q       # 26 tests: the frontend and the decoder's attention
```

Data prep also needs [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io) 3.4.3
(conda env `mfa`), `ffmpeg`, and Opus's `fargan_demo` (step 1). Large files live in gitignored folders:
`data/` (corpora, packs, runs), `models/` (PiVo and `fargan_demo`) and `results/`.

## Step 0: the frontend (text to phones)

`frontend.py` turns text into 74 symbols: Tamil phones, doubled consonants, word breaks and punctuation.

```bash
uv run python frontend.py "நான் meeting-க்கு 10.30-க்கு வருவேன்"
```

- Tamil is read by rule (stops voice between vowels and after nasals, `ற்ற` → tr, `ன்ற` → n dr, a
  word-final u is reduced). Latin-script English goes through CMUdict. Numbers are read in the
  spoken register with `uraikal.numbers`.
- Words it can't read: 0.20% on web text, 0.008% on literature, and none of IndicTTS's 9,437 sentences.
- Not done: romanized Tamil (spelled out letter by letter for now), times and dates, and a linguist's
  review of the rules.

## Step 1: the vocoder check

`resynth.py` extracts PiVo's 20 features from real speech with Opus's `fargan_demo` and plays them
back through PiVo's vocoder. It shows the best Kuyil can sound with this vocoder.

```bash
uv run python resynth.py in1.wav [in2.wav ...] --out results/fargan_check
```

- Local copies (gitignored) in `models/`:
  - `models/fargan/fargan_demo`: Opus at `bafd40d`, built with
    `./autogen.sh && ./configure --enable-deep-plc && make`.
  - `models/pivo-1.4m/`: `empero-ai/PiVo-1.4M` at revision `515d84f`.
- Measured on 10 Tamil clips: pitch within 3–6 cents, band energies 2.1–2.8 dB apart, and 3–18% of
  unvoiced frames come back voiced (lowest on studio audio).
- Only Opus's PyTorch FARGAN (`fargan_sq1Ab_adv_50.pth`) matches PiVo's vocoder. Opus's C synthesis
  uses different weights, so use `fargan_demo` for analysis only.
- Native listeners: `results/fargan_check/listen_pairs.wav` plays each clip, then its resynthesis.

## Step 2: training

PiVo's model is rebuilt from its report and its checkpoint's tensor shapes. At PiVo's sizes,
`model.py` has exactly PiVo's 584,965 acoustic parameters, and its timing signals match PiVo's
`phase.py` to 6e-8.

**Data, on a laptop** (IndicTTS Tamil, `SPRINGLab/IndicTTS_Tamil` at revision `6bd9a35`, in
`data/indictts_ta/`: 9,437 sentences, 2 speakers, 26.6 h):

1. `uv run python prepare.py data/indictts_ta data/indictts` makes 16 kHz audio, features and phones
   (about 4 min).
2. Align with Montreal Forced Aligner 3.4.3, the version PiVo used, trained on our phone set
   (about 40 min with 8 jobs):
   ```bash
   PATH=/opt/anaconda3/envs/mfa/bin:$PATH MFA_ROOT_DIR=$PWD/data/mfa_root mfa train data/indictts/corpus data/indictts/dictionary.txt data/indictts/mfa_model.zip --config_path mfa_train.yaml --output_directory data/indictts/aligned -j 8 --clean
   ```
3. `uv run python pack.py data/indictts data/indictts.pt` makes one training file (351 MB).

**Training** (`train.py`): PiVo's supervised recipe. Smooth-L1 on normalized features plus
0.1 × squared error on log(1 + duration), batch 8, learning rate 3e-4 with 200 warm-up updates,
90 epochs. A laptop GPU manages about 0.4 s a step, so full runs belong on a GPU server.

**On a GPU server** (it needs only Python and PyTorch ≥ 2.1). Set `POD` to the ssh command for the box
(`ssh -i KEY -p PORT`) and `HOST` to `user@address`. On a RunPod network volume use `rsync -rlt`,
because `-a` fails when it tries to change file owners there.

```bash
${=POD} $HOST 'mkdir -p /workspace/tamil-tts/kuyil && cd /workspace/tamil-tts && python3 -m venv --system-site-packages .venv'
rsync -rlt -e "$POD" model.py train.py run_ladder.sh data/indictts.pt $HOST:/workspace/tamil-tts/kuyil/
${=POD} $HOST 'cd /workspace/tamil-tts/kuyil && PYTHON=../.venv/bin/python nohup bash run_ladder.sh indictts.pt > ladder.log 2>&1 < /dev/null &'
${=POD} $HOST 'tail -qn 1 /workspace/tamil-tts/kuyil/runs/*.log'              # progress: one JSON line per epoch
rsync -rlt -e "$POD" $HOST:/workspace/tamil-tts/kuyil/runs/ data/runs/        # afterwards: checkpoints back
```

- `run_ladder.sh` trains 3 sizes with 3 seeds each, all at once on one GPU: 1.4M, 4.9M and 22.7M
  parameters in total with the vocoder. On an H100 PCIe that takes 175 / 211 / 326 s an epoch, about
  8 h for the ladder.
- The decoder looks back 64 frames. Its attention runs block by block (`local_attention`, checked
  against a dense mask in `test_model.py`); the dense version was 11× slower.
- When training ends, `run_ladder.sh` copies `runs/` to Cloudflare R2 and verifies it by checksum, if
  the box has an rclone remote `r2` (`R2_DEST` overrides the destination).
- Disk on the server: about 1 GB (pack 0.35 GB, 18 checkpoints about 0.63 GB), plus PyTorch.

## Step 3: listening and scoring

```bash
uv run python synth.py data/runs/<run>/best.pt "வணக்கம், நான் குயில்." --voice female --out hello.wav
uv run python synth.py data/runs/<run>/best.pt --test data/indictts --out data/eval/<run>
```

The second form speaks the 125 held-out sentences and writes speech-recognition manifests for both
the synthesized and the original audio. Run them through oosai's `eval/run_asr.py`: the gap between
the two error rates is the test. Native listeners then pick a size.

Results (2026-11-04, IndicConformer-600M on the 125 held-out sentences; character error rate, mean of 3 seeds):

| Audio | Character error | Word error |
|---|---|---|
| Original recordings | 2.9% | 19.2% |
| Kuyil 1.4M | 3.0% | 18.1% |
| Kuyil 5M | 3.0% | 18.2% |
| Kuyil 20M | 2.9% | 18.2% |

Every size is as intelligible as the recordings, so the recognizer can't pick a size; listeners will.
Synthesis runs at a real-time factor of 0.06–0.07 on one Mac CPU thread at every size, since the vocoder
dominates. 20M reaches its best validation loss by epoch 21 and then overfits 26 h of speech.

## What's not done

- PiVo's 1,200-update adversarial refinement.
- Confirmation from IIT Madras. The weights are released under CC BY 4.0, matching the
  `SPRINGLab/IndicTTS_Tamil` mirror they were trained from, and keep IIT Madras's IndicTTS copyright
  notice. IIT Madras's own licence permits free distribution of derivative works, but one clause on
  resale is ambiguous. The training pack holds IndicTTS features, so it never goes to a public hub.
- More voices and Tanglish speech in the training data.

## Licence

Code: Apache 2.0 ([LICENSE](LICENSE)). Released Kuyil weights: CC BY 4.0 (each model's `NOTICE`).
PiVo's weights and code are Apache 2.0, and FARGAN is under the Xiph/Opus BSD licence. Both are
downloaded separately into `models/` and keep their own notices. A released folder carries an
unchanged copy of PiVo's vocoder, with those notices, in `pivo/`. `release.py` builds the released folders.
Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).
