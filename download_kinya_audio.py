import os
import csv
import io
import soundfile as sf
from datasets import load_dataset, Audio

print("📥 Loading dataset...")
dataset = load_dataset("badrex/kinyarwanda-speech-sample", split="train")

# Disable automatic decoding to avoid torchcodec
dataset = dataset.cast_column("audio", Audio(decode=False))

os.makedirs("data/audio", exist_ok=True)
rows = []

print("🔄 Processing 100 samples...")

for i in range(100):
    sample = dataset[i]

    audio_bytes = sample["audio"]["bytes"]
    waveform, sample_rate = sf.read(io.BytesIO(audio_bytes))

    filename = f"sample_{i+1}.wav"
    filepath = os.path.join("data/audio", filename)

    sf.write(filepath, waveform, sample_rate)

    rows.append({
        "filename": filename,
        "transcript": sample["transcription"]
    })

print("📝 Writing CSV...")

with open("audio_transcripts.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "transcript"])
    writer.writeheader()
    writer.writerows(rows)

print("✅ DONE! Audio + transcripts ready.")
