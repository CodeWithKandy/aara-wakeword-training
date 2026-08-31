"""Downloads RIR (room reverb) and background music data for training augmentation."""
import os
from pathlib import Path
import numpy as np
import scipy.io.wavfile
import datasets


def save_dataset_to_wavs(dataset, output_dir: str, limit: int | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for i, row in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        audio = row["audio"]["array"]
        wav_path = os.path.join(output_dir, f"{i}.wav")
        scipy.io.wavfile.write(wav_path, 16000, (audio * 32767).astype(np.int16))
        count += 1
        if count % 100 == 0:
            print(f"  {output_dir}: {count} files written")
    print(f"  {output_dir}: {count} files written (done)")


print("Downloading MIT environmental impulse responses (RIRs)...")
rir_dataset = datasets.load_dataset(
    "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
).cast_column("audio", datasets.Audio(sampling_rate=16000))
save_dataset_to_wavs(rir_dataset, "mit_rirs")

print("Downloading FMA-small sample (background music)...")
fma_dataset = datasets.load_dataset(
    "rudraml/fma", name="small", split="train", trust_remote_code=True
).cast_column("audio", datasets.Audio(sampling_rate=16000))
save_dataset_to_wavs(fma_dataset, "fma", limit=2000)

print("All augmentation data ready.")