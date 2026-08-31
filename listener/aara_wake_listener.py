"""AARA wake word listener - continuous 'Hey AARA' detection."""
import collections
import queue
import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

MODEL_PATH = "model/hey_aara.onnx"
CHUNK_SIZE = 1280
SAMPLE_RATE = 16000
THRESHOLD = 0.5
WINDOW_FRAMES = 16
COOLDOWN_SECONDS = 1.5

model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
model_name = list(model.models.keys())[0]
model.reset()

audio_queue = queue.Queue()
recent_scores = collections.deque(maxlen=WINDOW_FRAMES)
last_trigger_time = 0


def callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


def on_wake_word_detected():
    """This is where AARA's actual response will eventually hook in
    (start STT, wake the assistant, etc.) - for now, just confirm it heard you."""
    print("\n🎙️  Hey AARA detected! Listening...\n")


print("=" * 50)
print("AARA wake word listener active - say 'Hey AARA'")
print("Press Ctrl+C to stop")
print("=" * 50)

with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype="float32", callback=callback):
    try:
        while True:
            chunk = audio_queue.get()
            audio_int16 = (chunk * 32767).astype(np.int16)
            score = model.predict(audio_int16)[model_name]
            recent_scores.append(score)
            window_max = max(recent_scores)

            now = time.time()
            if window_max >= THRESHOLD and (now - last_trigger_time) > COOLDOWN_SECONDS:
                on_wake_word_detected()
                last_trigger_time = now
    except KeyboardInterrupt:
        print("\nStopped.")