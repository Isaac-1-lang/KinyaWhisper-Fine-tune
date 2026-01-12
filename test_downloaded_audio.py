# #!/usr/bin/env python3
# """
# test_downloaded_audio.py

# Tests the KinyaWhisper model on the downloaded audio files from download_kinya_audio.py.
# Compares model predictions with ground truth transcripts and calculates accuracy metrics.
# """

# import csv
# import os
# from pathlib import Path
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torchaudio
# import torch
# from jiwer import wer, cer

# # ====== Configuration ======
# audio_dir = "data/audio"
# csv_file = "audio_transcripts.csv"
# output_file = "test_results.csv"
# model_name = "benax-rw/KinyaWhisper"  # Use Hugging Face model

# # ====== Load model and processor ======
# print("🔄 Loading KinyaWhisper model...")
# model = WhisperForConditionalGeneration.from_pretrained(model_name)
# processor = WhisperProcessor.from_pretrained(model_name)
# model.eval()
# print("✅ Model loaded successfully!")

# # ====== Load ground truth transcripts from CSV ======
# print(f"\n📖 Loading transcripts from {csv_file}...")
# ground_truth = {}
# with open(csv_file, "r", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         ground_truth[row["filename"]] = row["transcript"]

# print(f"✅ Loaded {len(ground_truth)} transcripts")

# # ====== Process each audio file ======
# results = []
# predictions = []
# references = []

# print(f"\n🎤 Transcribing audio files from {audio_dir}...")

# for i, (filename, true_transcript) in enumerate(ground_truth.items(), 1):
#     audio_path = Path(audio_dir) / filename
    
#     if not audio_path.exists():
#         print(f"⚠️  Skipping {filename} (file not found)")
#         continue
    
#     try:
#         # Load and preprocess audio
#         waveform, sample_rate = torchaudio.load(str(audio_path))
        
#         # Convert stereo to mono if needed
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0)
#         else:
#             waveform = waveform.squeeze()
        
#         # Prepare input
#         inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        
#         # Generate transcription
#         with torch.no_grad():
#             predicted_ids = model.generate(inputs["input_features"])
        
#         # Decode transcription
#         predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
#         # Store results
#         results.append({
#             "filename": filename,
#             "ground_truth": true_transcript,
#             "prediction": predicted_text,
#             "match": true_transcript.strip().lower() == predicted_text.strip().lower()
#         })
        
#         predictions.append(predicted_text)
#         references.append(true_transcript)
        
#         if i % 10 == 0:
#             print(f"  Processed {i}/{len(ground_truth)} files...")
            
#     except Exception as e:
#         print(f"❌ Error processing {filename}: {e}")
#         continue

# print(f"✅ Processed {len(results)} audio files!")

# # ====== Calculate metrics ======
# print("\n📊 Calculating accuracy metrics...")

# # Calculate Word Error Rate (WER) and Character Error Rate (CER)
# if predictions and references:
#     wer_score = wer(references, predictions)
#     cer_score = cer(references, predictions)
    
#     # Calculate exact match accuracy
#     exact_matches = sum(1 for r in results if r["match"])
#     accuracy = exact_matches / len(results) * 100 if results else 0
    
#     print(f"\n📈 Results:")
#     print(f"   Word Error Rate (WER): {wer_score:.2%}")
#     print(f"   Character Error Rate (CER): {cer_score:.2%}")
#     print(f"   Exact Match Accuracy: {accuracy:.2f}% ({exact_matches}/{len(results)})")

# # ====== Save results to CSV ======
# print(f"\n💾 Saving results to {output_file}...")
# with open(output_file, "w", newline="", encoding="utf-8") as f:
#     fieldnames = ["filename", "ground_truth", "prediction", "match"]
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(results)

# print(f"✅ Results saved to {output_file}")

# # ====== Print sample results ======
# print(f"\n📝 Sample Results (first 5):")
# print("-" * 80)
# for i, result in enumerate(results[:5], 1):
#     print(f"\n{i}. {result['filename']}")
#     print(f"   Ground Truth: {result['ground_truth']}")
#     print(f"   Prediction:   {result['prediction']}")
#     print(f"   Match: {'✅' if result['match'] else '❌'}")

# print("\n🎉 Testing complete!")


from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned Openai whisper-small model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("data/audio/sample_2.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)

