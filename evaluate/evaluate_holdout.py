"""Test the trained hey_aara.onnx model against our held-out test clips."""
from pathlib import Path
import numpy as np
import soundfile as sf
from openwakeword.model import Model

MODEL_PATH = "model/hey_aara.onnx"
POSITIVE_TEST_DIR = "training_output/hey_aara/positive_test"
NEGATIVE_TEST_DIR = "training_output/hey_aara/negative_test"
THRESHOLD = 0.5

model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
model_name = list(model.models.keys())[0]


def score_clip(path):
    audio, sr = sf.read(path, dtype="int16")
    model.reset()
    scores = []
    for i in range(0, len(audio), 1280):
        chunk = audio[i:i + 1280]
        if len(chunk) < 1280:
            break
        prediction = model.predict(chunk)
        scores.append(prediction[model_name])
    return max(scores) if scores else 0.0


def evaluate(directory, label):
    files = sorted(Path(directory).glob("*.wav"))
    scores = np.array([score_clip(f) for f in files])
    detected = (scores >= THRESHOLD).sum()
    print(f"{label}: {len(files)} clips, {detected} scored >= {THRESHOLD} "
          f"({100 * detected / len(files):.1f}%), mean score {scores.mean():.3f}")
    return scores


print("Positive clips (want HIGH detection rate):")
pos_scores = evaluate(POSITIVE_TEST_DIR, "  Positive")

print("Negative/adversarial clips (want LOW false-trigger rate):")
neg_scores = evaluate(NEGATIVE_TEST_DIR, "  Negative")