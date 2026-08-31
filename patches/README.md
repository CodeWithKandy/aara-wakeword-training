# openWakeWord patches

This project trains a custom wake word with [openWakeWord](https://github.com/dscripka/openWakeWord)
(David Scripka, Apache-2.0). openWakeWord is **not** vendored here — it is a real
third-party library. We clone it, check out a known commit, and apply
`openwakeword.patch` on top.

The patch fixes genuine bugs and adds one training feature. Every hunk is
explained below.

## Base revision

```
repo:   https://github.com/dscripka/openWakeWord.git
commit: 368c03716d1e92591906a84949bc477f3a834455
        ("put onnx to tflite conversion behind flag in training code", 2025-12-30)
```

## Applying the patch

```bash
git clone https://github.com/dscripka/openWakeWord.git openwakeword-src
cd openwakeword-src
git checkout 368c03716d1e92591906a84949bc477f3a834455
git apply ../patches/openwakeword.patch
pip install -e .
```

To regenerate the patch after further local edits:

```bash
git -C openwakeword-src diff HEAD -- openwakeword/data.py openwakeword/train.py openwakeword/utils.py \
  > patches/openwakeword.patch
```

## What each change does and why

### 1. `data.py` — route audio I/O through `soundfile` instead of `torchcodec`

Recent `torchaudio` moved `torchaudio.load()` / `torchaudio.info()` onto
`torchcodec`, which fails to load its native library on Windows with recent
CUDA builds (unresolved upstream). `soundfile` was already a reliable
dependency everywhere else in this pipeline, so the patch monkey-patches both
functions at import time to read via `soundfile` and rebuild the return types
`torchaudio` callers expect (a `(waveform_tensor, sample_rate)` tuple and an
`AudioMetaData`-shaped namedtuple).

This is the same class of problem that forces `datasets==3.6.0` in
`requirements.txt` — anything newer also drags in `torchcodec`.

### 2. `data.py` — release memory-mapped file handles before delete/rename (`trim_mmap`)

`trim_mmap()` writes a trimmed copy of a `.npy` feature file, deletes the
original, and renames the copy into place. On Linux the open `np.memmap`
objects don't block this. On Windows the OS holds a lock on a memory-mapped
file until every handle is dropped, so `os.remove()` / `os.rename()` raise
`PermissionError`. The patch explicitly `del`s `mmap_file1` before the delete
and `mmap_file2` before the rename, with a `gc.collect()` after each to force
the handle release.

### 3. `utils.py` — release the memmap handle in `compute_features_from_generator`

Same Windows constraint, one level up. `compute_features_from_generator()`
fills a memmap `fp`, then calls `trim_mmap()` on the same path. `fp` has to be
`del`'d + `gc.collect()`'d first, or `trim_mmap` can't replace the file.

### 4. `train.py` — add a `real_voice` positive training category

The synthetic-only model generalised poorly to a real microphone (13% recall
on real recordings). Adding a small set of real "hey AARA" recordings as a
separately-batched **positive** category closed most of that gap (→ 76.7%).
openWakeWord's trainer has no built-in hook for an extra labelled category, so
the patch touches three coordinated spots:

* **Feature generation** — after the positive/negative clips, augment the
  `real_voice_train/` WAVs (each repeated 30×, run through the same RIR +
  background-noise augmentation) and write `real_voice_features_train.npy`.
* **Label transform** — add `real_voice` to the label-transform loop and map it
  to label `1` (`if key in ("positive", "real_voice")`), alongside the existing
  `positive` handling.
* **Feature-file wiring** — register
  `config["feature_data_files"]["real_voice"]` so the batch generator picks it
  up. Its batch size comes from `batch_n_per_class.real_voice` in the YAML.

### 5. `train.py` — force `DataLoader(num_workers=0)`

The training pipeline includes a lambda in the transform chain. On Windows,
`DataLoader(num_workers > 0)` tries to pickle it for the worker processes and
raises `PicklingError`. Forcing `num_workers=0` (single-process loading) is the
straightforward fix; throughput was not a bottleneck for this dataset size.

## Known upstream bug (worked around, not patched)

`train.py`'s argparse flags are declared with `default="False"` — the **string**
`"False"`, which is always truthy. Only `if arg is True` checks behave
correctly; a plain `if arg:` check (e.g. the TFLite-conversion path) fires even
when the flag wasn't passed. We avoid this by never relying on those flags
(TFLite export is skipped entirely — see the main README).
