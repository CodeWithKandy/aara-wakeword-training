"""Resamples all positive/negative train+test clips to 16kHz, in place."""
from pathlib import Path
import librosa
import soundfile as sf

TARGET_SR = 16000
DIRS = [
    r"training_output\hey_aara\positive_train",
    r"training_output\hey_aara\positive_test",
    r"training_output\hey_aara\negative_train",
    r"training_output\hey_aara\negative_test",
]

for d in DIRS:
    wav_files = list(Path(d).glob("*.wav"))
    print(f"{d}: {len(wav_files)} files")
    fixed = 0
    for i, wav_path in enumerate(wav_files):
        info = sf.info(str(wav_path))
        if info.samplerate != TARGET_SR:
            audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
            sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_16")
            fixed += 1
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}/{len(wav_files)} checked")
    print(f"  {d}: resampled {fixed} files")

print("Done.")
