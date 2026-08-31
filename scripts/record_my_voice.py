"""Record real 'hey AARA' clips for validation and potential fine-tuning."""
import os
import time
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
DURATION = 2.5
N_CLIPS = 40
OUTPUT_DIR = "my_voice_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(N_CLIPS):
    input(f"Press Enter when ready for clip {i + 1}/{N_CLIPS}...")
    print("  Get ready...")
    time.sleep(1.0)
    print("  Recording NOW - say 'hey AARA'!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    path = f"{OUTPUT_DIR}/clip_{i:02d}.wav"
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"  saved {path}, peak={abs(audio).max():.3f}")

print("Done recording.")