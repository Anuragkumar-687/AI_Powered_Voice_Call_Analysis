import streamlit as st
import whisper
import torch
import json
import os
from transformers import pipeline

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

st.title("AI-Powered Voice Call Analysis")

st.write("Upload an audio file to analyze customer-agent conversations.")

# Load Whisper Model
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

# Load Phi-3 Mini Model
@st.cache_resource
def load_phi3():

    pipe = pipeline(
        "text-generation",
        model="microsoft/Phi-3-mini-4k-instruct",
        device_map="cpu",
        model_kwargs={"torch_dtype": torch.float32}
    )

    return pipe

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "m4a"]
)

if uploaded_file is not None:

    # Save Uploaded Audio
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Whisper Transcription
    st.write("Loading Whisper model...")
    whisper_model = load_whisper()

    st.write("Transcribing audio...")

    result = whisper_model.transcribe(
        "temp_audio.wav",
        language="hi",
        task="translate",
        fp16=False,
        temperature=0
    )

    transcript = result["text"]

    # Raw Transcript
    st.subheader("Raw Transcript")
    st.write(transcript)

    # Format Transcript
    lines = transcript.split(".")

    formatted_transcript = ""

    for i, line in enumerate(lines):

        line = line.strip()

        if line:

            speaker = "Agent" if i % 2 == 0 else "Customer"

            formatted_transcript += f"{speaker}: {line}\n"

    # Display Formatted Transcript
    st.subheader("Formatted Transcript")
    st.text(formatted_transcript)

    # Load Phi-3
    st.write("Loading Phi-3 Mini model...")
    pipe = load_phi3()

    # Prompt
    prompt = f"""
You are an AI Call Quality Analyst.

Analyze the following customer support conversation.

Return STRICT JSON output with:
1. call_summary
2. sentiment
3. agent_score_out_of_10
4. strengths
5. improvement_areas
6. recommended_next_steps

Transcript:
{formatted_transcript}

STRICT JSON ONLY.
"""

    # Generate Analysis
    st.write("Generating AI analysis...")

    try:

        response = pipe(
            prompt,
            max_new_tokens=150
        )

        output = response[0]["generated_text"]

        start = output.find("{")
        end = output.rfind("}") + 1

        json_output = output[start:end]

        data = json.loads(json_output)

        # JSON Report
        st.subheader("AI Analysis Report (JSON)")
        st.json(data)

        # Ensure List Format
        if isinstance(data["strengths"], str):
            data["strengths"] = [data["strengths"]]

        if isinstance(data["improvement_areas"], str):
            data["improvement_areas"] = [data["improvement_areas"]]

        if isinstance(data["recommended_next_steps"], str):
            data["recommended_next_steps"] = [data["recommended_next_steps"]]

        # Markdown Report
        markdown_report = f"""
# AI Voice Call Analysis Report

## Call Summary
{data["call_summary"]}

## Sentiment
{data["sentiment"]}

## Agent Score
{data["agent_score_out_of_10"]}/10

## Strengths
• {" ".join(data["strengths"])}

## Improvement Areas
• {" ".join(data["improvement_areas"])}

## Recommended Next Steps
• {" ".join(data["recommended_next_steps"])}
"""

        # Display Markdown Report
        st.subheader("Readable Markdown Report")
        st.markdown(markdown_report)

    except Exception as e:

        st.error("Failed to generate AI analysis.")

        st.write(str(e))