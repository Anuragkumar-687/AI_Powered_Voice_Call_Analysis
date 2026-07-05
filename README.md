# 🎧 AI-Powered Voice Call Analysis Using Open-Source SLMs

An end-to-end AI pipeline that analyzes customer support calls and generates structured coaching feedback using open-source models.

---

## 🚀 Overview

This project processes recorded customer support calls and transforms them into actionable insights.

It uses **Whisper** for speech-to-text transcription and **Phi-3 Mini (SLM)** for intelligent conversation analysis, producing structured and human-readable reports for agent performance improvement.

---

## ✨ Key Features

- 🎙️ Supports audio formats: `.mp3`, `.wav`, `.m4a`
- 🧠 Whisper-based speech-to-text transcription
- 🌐 Hindi → English transcription support
- 👥 Speaker separation (Agent vs Customer formatting)
- 🤖 AI-powered analysis using Phi-3 Mini
- 📊 Structured JSON report generation
- 📝 Human-readable Markdown report
- ⚡ Lightweight and runs on Google Colab

---


## Technologies Used

- Python
- Whisper
- Phi-3 Mini
- Hugging Face Transformers
- Google Colab

---

## Project Workflow

Audio Input  
↓  
Whisper Speech-to-Text  
↓  
Transcript Formatting  
↓  
Phi-3 Mini Analysis  
↓  
JSON + Markdown Report Generation

---

## Output Files

- report.json
- report.md

---

## How to Run

1. Install dependencies from requirements.txt
2. Open the notebook in Google Colab
3. Upload audio file
4. Run notebook cells sequentially
5. Generated reports will be saved automatically

---

## Model Used

### Phi-3 Mini 4K Instruct

Reason for selection:
- Open-source
- Lightweight
- Efficient on Google Colab
- Good instruction-following capability
- Reliable structured JSON output generation

---

## Author

Anurag Kumar
