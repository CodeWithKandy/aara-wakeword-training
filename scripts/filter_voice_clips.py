"""Copy only the clean (non-silent) recordings into their own training directory."""
import shutil
from pathlib import Path
import soundfile as sf
import numpy as np

MIN_PEAK = 0.1
src = Path("my_voice_samples")
dst = Path("training_output/hey_aara/real_voice_train")

count = 0
for path in sorted(src.glob("*.wav")):
    audio, sr = sf.read(path, dtype="float32")
    if np.abs(audio).max() >= MIN_PEAK:
        shutil.copy(path, dst / path.name)
        count += 1
print(f"Copied {count} clean clips to {dst}")