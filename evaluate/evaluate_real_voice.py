"""Test the trained model against real recorded 'hey AARA' clips (usable ones only)."""
from pathlib import Path
import numpy as np
import soundfile as sf
from openwakeword.model import Model

MODEL_PATH = "model/hey_aara.onnx"
CLIPS_DIR = "my_voice_samples"
MIN_PEAK = 0.1

model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
model_name = list(model.models.keys())[0]

results = []
for path in sorted(Path(CLIPS_DIR).glob("*.wav")):
    audio, sr = sf.read(path, dtype="float32")
    if np.abs(audio).max() < MIN_PEAK:
        continue  # skip silent/bad recordings
    audio_int16 = sf.read(path, dtype="int16")[0]
    model.reset()
    scores = []
    for i in range(0, len(audio_int16), 1280):
        chunk = audio_int16[i:i + 1280]
        if len(chunk) < 1280:
            break
        scores.append(model.predict(chunk)[model_name])
    max_score = max(scores) if scores else 0.0
    detected = max_score >= 0.5
    results.append(detected)
    print(f"{path.name}: max_score={max_score:.3f}  [{'DETECTED' if detected else 'missed'}]")

print(f"\n{sum(results)}/{len(results)} detected ({100*sum(results)/len(results):.1f}%)")