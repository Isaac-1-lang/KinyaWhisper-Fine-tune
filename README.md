## 🗣️ KinyaWhisper
KinyaWhisper is a fine-tuned version of OpenAI's Whisper model for automatic speech recognition (ASR) in Kinyarwanda. It was trained on 102 manually labeled .wav files and serves as a reproducible baseline for speech recognition in low-resource, indigenous languages.

## 🚀 Quick Start

**Transcribe an audio file (fastest way):**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit inference.py to set your audio file path
# 3. Run inference
python inference.py
```

**Use the model in your code:**
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## 🤗 Hugging Face Model

The fine-tuned KinyaWhisper model is publicly available on Hugging Face:

➡️ [https://huggingface.co/benax-rw/KinyaWhisper](https://huggingface.co/benax-rw/KinyaWhisper)

---

## 📖 Usage Guide

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have Python 3.7+ installed**

### 1. 🎤 Using the Model for Inference (Transcription)

#### Option A: Use the provided `inference.py` script

The `inference.py` script transcribes a single audio file and saves the result to a text file.

**Steps:**
1. Place your audio file in the `unseen_audio_data/` directory (or update the path in the script)
2. Edit `inference.py` to set your audio file path:
   ```python
   audio_path = "unseen_audio_data/your_audio.mp3"  # or .wav
   ```
3. Run the script:
   ```bash
   python inference.py
   ```
4. The transcription will be:
   - Printed to the console
   - Saved to `transcription_output/` directory as a `.txt` file

#### Option B: Use the model directly in your code

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned KinyaWhisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)
```

### 2. 🧪 Testing the Model on Training Data

The `testing.py` script tests the model on all audio files in the `audio/` directory (the training dataset).

**Steps:**
1. Ensure the model files are in `kinya-whisper-model/` directory (or update the path in the script)
2. Run:
   ```bash
   python testing.py
   ```
3. Results:
   - Transcriptions are printed to the console
   - All transcriptions are saved to `transcriptions.txt`

### 3. 📊 Evaluating Model Performance (WER)

The `wer.py` script calculates the Word Error Rate (WER) by comparing predictions with ground truth.

**Steps:**
1. First, run `testing.py` to generate `transcriptions.txt`
2. Run:
   ```bash
   python wer.py
   ```
3. The script will display the WER percentage

### 4. 🏋️ Fine-tuning/Retraining the Model

The `train.py` script fine-tunes the Whisper model on your custom dataset.

**Steps:**
1. Prepare your dataset:
   - Place audio files in the `audio/` directory
   - Create or update `dataset.jsonl` with format:
     ```json
     {"audio": "audio/001.wav", "text": "transcription text"}
     {"audio": "audio/002.wav", "text": "another transcription"}
     ```
2. Configure training:
   - Edit `train.py` to adjust hyperparameters (learning rate, batch size, epochs, etc.)
   - For first-time training, change the model path from `./kinya-whisper-model` to `"openai/whisper-small"`
3. Run training:
   ```bash
   python train.py
   ```
4. The trained model will be saved to `./kinya-whisper-model/`

**Training Configuration (current settings):**
- Model: `openai/whisper-small`
- Epochs: 80
- Batch size: 4
- Learning rate: 1e-5
- Optimizer: Adam

---

## 📁 Project Structure

```
KinyaWhisper-Fine-tune/
├── audio/                    # Training audio files (102 .wav files)
├── unseen_audio_data/        # Audio files for inference
├── kinya-whisper-model/      # Fine-tuned model files
├── transcription_output/     # Output directory for transcriptions
├── dataset.jsonl             # Training dataset (audio paths + transcriptions)
├── train.py                  # Fine-tuning script
├── inference.py              # Single audio file transcription
├── testing.py                # Test model on training dataset
├── wer.py                    # Calculate Word Error Rate
├── transcriptions.txt        # Generated transcriptions (from testing.py)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🏋️ Taining Details
•	Model: openai/whisper-small
•	Epochs: 80
•	Batch size: 4
•	Learning rate: 1e-5
•	Optimizer: Adam
•	Final loss: 0.00024
•	WER: 51.85%

## ⚠️Limitations
The model was trained on a small dataset (102 samples). It performs best on short, clear Kinyarwanda utterances and may struggle with longer or noisy audio. This is an early-stage educational model, not yet suitable for production use.

## 📚 Citation

If you use this model, please cite:

```bibtex
@misc{baziramwabo2025kinyawhisper,
  author       = {Gabriel Baziramwabo},
  title        = {KinyaWhisper: Fine-Tuning Whisper for Kinyarwanda ASR},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/benax-rw/KinyaWhisper}},
  note         = {Version 1.0}
}
```
## 📬 Contact
Maintained by Gabriel Baziramwabo. 
✉️ gabriel@benax.rw
🔗 https://benax.rw
