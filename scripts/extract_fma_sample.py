"""Extracts and converts a random sample of FMA-small tracks to 16kHz WAV."""
import random
import tempfile
import zipfile
from pathlib import Path

import soundfile as sf
import librosa

ZIP_PATH = "fma_small.zip"
OUTPUT_DIR = "fma"
N_SAMPLES = 2000
SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Opening {ZIP_PATH}...")
with zipfile.ZipFile(ZIP_PATH) as zf:
    all_mp3s = [n for n in zf.namelist() if n.lower().endswith(".mp3")]
    print(f"Found {len(all_mp3s)} tracks in archive.")

    random.seed(SEED)
    selected = random.sample(all_mp3s, min(N_SAMPLES, len(all_mp3s)))

    with tempfile.TemporaryDirectory() as tmp_dir:
        converted = 0
        for i, member in enumerate(selected):
            try:
                extracted_path = zf.extract(member, tmp_dir)
                audio, sr = librosa.load(extracted_path, sr=16000, mono=True)
                sf.write(Path(OUTPUT_DIR) / f"{i}.wav", audio, 16000, subtype="PCM_16")
                converted += 1
            except Exception as e:
                print(f"  skipping {member}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(selected)} processed")

print(f"Done. {converted} FMA clips converted to {OUTPUT_DIR}/")