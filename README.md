# aara-wakeword-training

Training pipeline, evaluation scripts, and a working live listener for a custom
**"Hey AARA"** wake word, built on [openWakeWord](https://github.com/dscripka/openWakeWord).

This repo is the wake-word work extracted from the larger AARA assistant project
so it can stand on its own. The trained model (`model/hey_aara.onnx`, ~215 KB) is
checked in — you can run the listener immediately without retraining.

---

## What's here

```
aara-wakeword-training/
├── config/hey_aara.yaml      training config (phrase, sample counts, batch mix, model size)
├── scripts/                  data prep: augmentation download, FMA extraction,
│                             resampling, voice recording, screening/filtering
│   └── generate_samples_stub/  no-op shim so openWakeWord's train.py imports cleanly
│                               when synthetic clips are generated out-of-band
├── patches/                  documented diff of our openWakeWord fixes (see patches/README.md)
├── evaluate/                 evaluate_holdout.py, evaluate_real_voice.py, threshold_sweep.py
├── listener/aara_wake_listener.py   the working live "Hey AARA" listener
└── model/hey_aara.onnx       the trained model
```

Large/generated inputs (training data, downloaded models, `.npy` features, raw
WAV folders, the venv, and the cloned third-party sources) are **not** tracked —
see `.gitignore`. You recreate them locally with the setup + pipeline steps below.

---

## How it works

openWakeWord trains a small DNN classifier on top of a fixed speech-embedding
model. The training mix for this wake word:

| Category | Count | Label | Source |
|---|---:|:---:|---|
| Synthetic positives ("hey AARA") | 20,000 | 1 | Piper TTS (LibriTTS-R voices) |
| Synthetic adversarial negatives | 19,500 | 0 | 15 confusable phrases: "hey Sara/Cara/Arya/Alexa/Siri/Google", "okay AARA", "hey there", … |
| Generic negative audio | ~2,000 hrs | 0 | openWakeWord's precomputed ACAV100M features |
| **Real-voice positives** | 30 clips × 30 | 1 | Recorded on a real mic, augmented (see below) |

The **real-voice category is our addition** to openWakeWord's trainer. A
synthetic-only model scored ~13% recall on real microphone recordings; adding a
small, separately-batched set of real "hey AARA" clips (each repeated 30× through
RIR + background-noise augmentation) brought held-out real-voice recall to
**76.7%**. See `patches/README.md` §4 for the code change.

Augmentation uses 270 MIT room impulse responses and ~2,000 FMA background-music
clips.

---

## Setup

Requires Python 3.10–3.12. Windows notes are called out; the pipeline was
developed and run on Windows 11 + CUDA 13.0.

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### 1. openWakeWord (patched — install from source)

Do **not** `pip install openwakeword`. Our patch fixes a torchaudio/torchcodec
incompatibility, Windows memory-map file-lock bugs, and adds the real-voice
training category.

```bash
git clone https://github.com/dscripka/openWakeWord.git openwakeword-src
cd openwakeword-src
git checkout 368c03716d1e92591906a84949bc477f3a834455
git apply ../patches/openwakeword.patch
pip install -e .
cd ..
```

### 2. piper-sample-generator (synthetic speech — install from source)

The PyPI package omits `piper_train` and the model `.pt.json` sidecars.

```bash
git clone https://github.com/rhasspy/piper-sample-generator.git piper-sample-generator-src
pip install -e piper-sample-generator-src
```

Download a Piper voice (LibriTTS-R high) into `models/`:

```bash
mkdir -p models
# en-us-libritts-high.pt  + en-us-libritts-high.pt.json
# from https://github.com/rhasspy/piper-sample-generator/releases
```

### 3. Precomputed negative features (from openWakeWord)

Download into `data/negative/features/`:

- `openwakeword_features_ACAV100M_2000_hrs_16bit.npy` (~17 GB) — generic negatives
- `validation_set_features.npy` — false-positive validation set

Both are published with openWakeWord's
[automatic model training docs](https://github.com/dscripka/openWakeWord/blob/main/docs/custom_model_training.md).

---

## Running the pipeline

### A. Augmentation data

```bash
python scripts/download_augmentation_data.py     # MIT RIRs -> mit_rirs/,  FMA music -> fma/
# or, if you already have fma_small.zip locally:
python scripts/extract_fma_sample.py
```

### B. Synthetic clips

Generate positive and adversarial-negative clips with piper-sample-generator into:

```
training_output/hey_aara/positive_train/     (20,000)
training_output/hey_aara/positive_test/      ( 4,000)
training_output/hey_aara/negative_train/     (adversarial, ~19,500)
training_output/hey_aara/negative_test/      (adversarial, ~3,900)
```

openWakeWord's `train.py --generate_clips` can do this automatically, but on this
setup generation was run separately (torchcodec issues, see patches), so
`config/hey_aara.yaml` points `piper_sample_generator_path` at the no-op stub and
`train.py` is only ever invoked **without** `--generate_clips`.

Then normalise everything to 16 kHz:

```bash
python scripts/resample_training_clips.py
```

### C. Real-voice clips

```bash
python scripts/record_my_voice.py        # records 40 "hey AARA" clips -> my_voice_samples/
python scripts/screen_voice_samples.py   # prints peak level of each (sanity check)
python scripts/filter_voice_clips.py      # copies non-silent clips -> training_output/hey_aara/real_voice_train/
```

### D. Features + training

```bash
python openwakeword-src/openwakeword/train.py --training_config config/hey_aara.yaml --augment_clips
python openwakeword-src/openwakeword/train.py --training_config config/hey_aara.yaml --train_model
```

Output: `training_output/hey_aara.onnx` (+ `.onnx.data`). Copy it to `model/` to
use it with the listener and evaluation scripts.

---

## Evaluation

Run from the repo root.

```bash
python evaluate/evaluate_holdout.py      # synthetic held-out positive + adversarial-negative clips
python evaluate/evaluate_real_voice.py   # real recordings in my_voice_samples/  -> the 76.7% number
python evaluate/threshold_sweep.py       # recall vs. false-accept at thresholds 0.3 / 0.5 / 0.7 / 0.9
```

### Results

| Test set | Metric | Result |
|---|---|---|
| Synthetic held-out | recall | ~72–76% |
| Synthetic held-out (confusable phrases) | false-accept | ~2–4% |
| **Real voice, held-out static recordings** | **recall** | **76.7% (23 / 30 clips)** |

Adding the real-voice training category is what moved real-voice recall from
13.3% → 76.7%.

---

## Live listener

```bash
python listener/aara_wake_listener.py
```

Says a line to the console on each detection. `on_wake_word_detected()` is the
integration hook point.

**Detection logic** (don't score single frames against a threshold — that path
was a long debugging detour):

1. `model.reset()` once after constructing `Model` — required for streaming.
2. Keep a rolling window of the last **16 frames** (~1.28 s) of scores.
3. Trigger when `max(window) >= 0.5`, not on the current frame alone.
4. Apply a **1.5 s cooldown** after a trigger so one utterance fires once.

---

## Known limitations

- **Live-streaming accuracy was never rigorously measured.** The 76.7% figure is
  from scoring static recordings offline. Live testing was qualitatively good but
  there is no counted live detection rate or false-accept-per-hour number.
- **Threshold (0.5) and cooldown (1.5 s) are working defaults, not validated
  optima.** `threshold_sweep.py` shows the trade-off on synthetic data; they were
  not tuned against real-world usage.
- **Windows-specific pipeline.** Fixes assume Windows file-locking and a
  torchcodec-free audio path. It should run on Linux with the same patch, but
  that hasn't been exercised.
- **`datasets` is pinned to 3.6.0.** Newer versions pull in torchcodec.
- **No TFLite export.** Deliberately skipped — needs `onnx_tf` +
  `tensorflow-cpu==2.8.1`, which predate Python 3.12. Not needed for ONNX Runtime
  deployment.
- **"AARA" alone (no "hey" prefix) is not supported.** A two-syllable bare name
  is far more false-positive-prone than a full phrase; adding it would need its
  own training run.

---

## Third-party components & licenses

- **openWakeWord** — Apache-2.0, © David Scripka. Used via `patches/`, not vendored.
- **piper-sample-generator / Piper** — MIT, © Michael Hansen.
- MIT environmental impulse responses; FMA (Free Music Archive) for background noise.

This repo's own code (scripts, config, evaluation, listener) is released under the
MIT License — see `LICENSE`.
