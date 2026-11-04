# Contributing to Kuyil

Thank you for helping Tamil speak. Issues and pull requests are welcome, in Tamil or English.

## Ways to help

- **Listen.** Native listeners are what decides between models. Rate the files a release asks about,
  or open an issue with text Kuyil says wrongly: the text, what it said, and what it should say.
- **Pronunciation rules.** `frontend.py` reads Tamil by rule. Linguists and careful speakers can check
  the rules and the expected phones in `test_frontend.py`.
- **Code.** Training, speed, the vocoder, and the items under "What's not done" in the README.

## Before a pull request

1. `uv sync`, then `uv run python -m pytest -q`: every test passes.
2. Keep the change small and in the style of the code around it. New logic gets a small test.
3. Say in the pull request what you measured, if the change affects speed or sound.

## Rules that protect people

- **Voices only with written consent.** Never train on, clone or imitate a real person's voice
  without that person's written consent.
- **No data or weights in git.** Corpora, packs, checkpoints and audio stay in the gitignored
  `data/`, `models/` and `results/` folders.
- **Test sets stay private.** Don't add, train on or publish held-out test sentences.
- **Say where data comes from.** Any new dataset needs its source, revision and licence in the README.

By contributing you agree your work is released under the Apache 2.0 licence ([LICENSE](LICENSE)).
