# 🎙️ VOICE-LINGUA V2

> A multilingual speech and text processing Streamlit web application built with Hugging Face Transformers, Google TTS, and Whisper.

---

## 🚀 Features

- 🎧 **Speech Recognition**  
  Upload audio files and transcribe them using OpenAI's Whisper.

- 🌍 **Translation**  
  Translate text from one language to another using Meta’s NLLB-200 model.

- 🗣️ **Speech Generation**  
  Generate speech (MP3 audio) from translated text using gTTS.

- 🎬 **Audio Extraction**  
  Extract audio from uploaded video files.

- ✂️ **Text Summarization**  
  Summarize long pieces of text using DistilBART.

---

## 🛠 Tech Stack

- **Frontend**: Streamlit  
- **ML Models**: Hugging Face Transformers  
  - `openai/whisper-large-v3` (ASR)  
  - `facebook/nllb-200-distilled-600M` (Translation)  
  - `sshleifer/distilbart-cnn-12-6` (Summarization)
- **Text-to-Speech**: `gTTS`  
- **Language Detection**: `langid`  
- **Video Processing**: `moviepy`  
- **Translation Backup**: `deep-translator`  


