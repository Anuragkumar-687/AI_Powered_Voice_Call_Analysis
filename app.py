import gradio as gr
import whisper
import torch
import json
from transformers import pipeline

# Load Whisper
whisper_model = whisper.load_model("tiny")

# Load Phi-3 Mini
pipe = pipeline(
    "text-generation",
    model="microsoft/Phi-3-mini-4k-instruct",
    device=-1
)

def analyze_call(audio_file):

    # Transcribe Audio
    result = whisper_model.transcribe(
        audio_file,
        fp16=False
    )

    transcript = result["text"]

    # Format Transcript
    lines = transcript.split(".")

    formatted_transcript = ""

    for i, line in enumerate(lines):

        line = line.strip()

        if line:

            speaker = "Agent" if i % 2 == 0 else "Customer"

            formatted_transcript += f"{speaker}: {line}\n"

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

    # Generate AI Analysis
    response = pipe(
        prompt,
        max_new_tokens=80
    )

    output = response[0]["generated_text"]

    start = output.find("{")
    end = output.rfind("}") + 1

    json_output = output[start:end]

    try:

        data = json.loads(json_output)

    except:

        data = {
            "error": "Failed to parse model output"
        }

    markdown_report = f"""
# AI Voice Call Analysis Report

## Formatted Transcript
{formatted_transcript}

## JSON Report
{json.dumps(data, indent=2)}
"""

    return markdown_report

# Gradio UI
interface = gr.Interface(
    fn=analyze_call,
    inputs=gr.Audio(type="filepath"),
    outputs="markdown",
    title="AI-Powered Voice Call Analysis",
    description="Upload a customer-agent call recording for AI analysis."
)

interface.launch()
