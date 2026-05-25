# AI-Powered Voice Call Analysis Using Open-Source SLMs

## Overview

This project is an AI-powered voice call analysis pipeline built using open-source technologies.

The system accepts a recorded customer support call, transcribes the audio using Whisper speech-to-text, and analyzes the conversation using the Phi-3 Mini Small Language Model (SLM) to generate structured coaching feedback for the call agent.

The generated output includes:
- Call Summary
- Sentiment Analysis
- Agent Score
- Strengths
- Improvement Areas
- Recommended Next Steps

---

## Features

- Audio ingestion (.mp3, .wav, .m4a)
- Whisper speech-to-text transcription
- Hindi-to-English transcription support
- Agent and Customer transcript formatting
- Phi-3 Mini conversational analysis
- Structured JSON report generation
- Human-readable Markdown report generation

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