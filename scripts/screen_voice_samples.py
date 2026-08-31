"""Screen recorded clips by actual signal level, flag which are usable."""
from pathlib import Path
import soundfile as sf
import numpy as np

MIN_PEAK = 0.1

for path in sorted(Path("my_voice_samples").glob("*.wav")):
    audio, sr = sf.read(path, dtype="float32")
    peak = np.abs(audio).max()
    status = "OK" if peak >= MIN_PEAK else "SILENT - discard"
    print(f"{path.name}: peak={peak:.3f}  [{status}]")